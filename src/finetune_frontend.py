import datetime
import math
import os
import random
import shutil
import time
import traceback
import builtins
from datetime import timedelta
from pathlib import Path
from typing import Sized
from contextlib import nullcontext

import hydra
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.logging import get_logger
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

import croco.utils.misc as misc
import dust3r.utils.path_to_croco  # noqa: F401
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler
from ovggt.losses.frontend_supervised import FrontendSupervisedLoss
from ovggt.models.ovggt import OVGGT
from train_frontend import (
    build_validation_loader,
    build_dataset,
    evaluate_frontend_epoch,
    freeze_frontend_stage_a_parameters,
    load_student_pretrained_weights,
    normalize_batch_images_,
    normalize_resume_step,
    save_current_code,
    should_run_validation,
    setup_for_distributed,
    summarize_trainable_parameters,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.multiprocessing.set_sharing_strategy("file_system")

printer = get_logger(__name__, log_level="DEBUG")


def supervised_loss_of_one_batch(
    batch,
    model,
    criterion,
    use_amp=False,
    use_activation_offload_cpu: bool = False,
):
    autocast_enabled = bool(use_amp) and torch.cuda.is_available()
    autocast_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    autocast_device_type = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.amp.autocast(
        device_type=autocast_device_type,
        enabled=autocast_enabled,
        dtype=autocast_dtype,
    ):
        activation_offload_enabled = (
            bool(use_activation_offload_cpu)
            and torch.cuda.is_available()
            and hasattr(torch.autograd, "graph")
            and hasattr(torch.autograd.graph, "save_on_cpu")
        )
        activation_ctx = (
            torch.autograd.graph.save_on_cpu(pin_memory=True)
            if activation_offload_enabled
            else nullcontext()
        )
        with activation_ctx:
            student_outputs = model(batch)
            if student_outputs.keyframe_schedule is None:
                raise RuntimeError("Frontend supervised training output is missing keyframe_schedule")
            student_outputs.views = None
            with torch.amp.autocast(device_type=autocast_device_type, enabled=False):
                loss = criterion(batch, student_outputs, student_outputs.keyframe_schedule)
    del student_outputs
    return loss


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Sized,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    epoch: int,
    loss_scaler,
    args,
    log_writer=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    accum_iter = args.accum_iter

    def save_model(epoch, fname, step):
        unwrapped_model = accelerator.unwrap_model(model)
        misc.save_model(
            accelerator=accelerator,
            args=args,
            model_without_ddp=unwrapped_model,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
            step=step,
            fname=fname,
        )

    if log_writer is not None:
        printer.info("log_dir: %s", log_writer.log_dir)

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    if (
        hasattr(data_loader, "batch_sampler")
        and hasattr(data_loader.batch_sampler, "batch_sampler")
        and hasattr(data_loader.batch_sampler.batch_sampler, "set_epoch")
    ):
        data_loader.batch_sampler.batch_sampler.set_epoch(epoch)

    optimizer.zero_grad()
    data_iter = metric_logger.log_every(data_loader, args.print_freq, accelerator, header)

    save_iter_freq = int(getattr(args, "save_iter_freq", 0) or 0)
    keep_iter_freq = int(getattr(args, "keep_iter_freq", 0) or 0)
    resume_step_raw = int(getattr(args, "start_step", 0) or 0)
    resume_epoch = int(getattr(args, "start_epoch", 0) or 0)
    resume_step = normalize_resume_step(resume_step_raw, len(data_loader))
    if resume_step != resume_step_raw and accelerator.is_main_process:
        printer.warning(
            "Normalizing resume step from %d to %d (steps_per_epoch=%d).",
            resume_step_raw,
            resume_step,
            len(data_loader),
        )

    for data_iter_step, batch in enumerate(data_iter):
        if epoch == resume_epoch and data_iter_step < resume_step:
            continue

        with accelerator.accumulate(model):
            normalize_batch_images_(batch)

            epoch_f = epoch + data_iter_step / len(data_loader)
            if data_iter_step % accum_iter == 0:
                misc.adjust_learning_rate(optimizer, epoch_f, args)

            step = int(epoch_f * len(data_loader))

            if data_iter_step % args.print_freq == 0 and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                printer.info(
                    "[Memory] Step %d: Allocated=%.2fGB, Reserved=%.2fGB",
                    step,
                    allocated,
                    reserved,
                )

            loss, loss_details = supervised_loss_of_one_batch(
                batch=batch,
                model=model,
                criterion=criterion,
                use_amp=bool(args.amp),
                use_activation_offload_cpu=bool(getattr(args, "activation_offload_cpu", False)),
            )
            loss_value = float(loss)

            if not math.isfinite(loss_value):
                printer.error("Loss is %s, stopping training. Details: %s", loss_value, loss_details)
                raise FloatingPointError(
                    f"Non-finite frontend supervised loss detected: loss={loss_value}, details={loss_details}"
                )

            loss_scaler(
                loss,
                optimizer,
                parameters=model.parameters(),
                update_grad=accelerator.sync_gradients,
                clip_grad=1.0,
            )
            current_lr = float(optimizer.param_groups[0]["lr"])
            metric_logger.update(epoch=epoch_f)
            metric_logger.update(lr=current_lr)
            metric_logger.update(step=step)
            metric_logger.update(loss=loss_value, **loss_details)
            if accelerator.sync_gradients and (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad(set_to_none=True)
                loss_value_reduce = accelerator.gather(
                    torch.tensor(loss_value, device=accelerator.device)
                ).mean()
                if log_writer is not None:
                    log_writer.add_scalar("train_loss", loss_value_reduce, step)
                    log_writer.add_scalar("train_lr", current_lr, step)
                    log_writer.add_scalar("train_iter", int(epoch_f * 1000), step)
                    for name, val in loss_details.items():
                        if isinstance(val, dict):
                            continue
                        if isinstance(val, torch.Tensor) and val.ndim > 0:
                            continue
                        log_writer.add_scalar("train_" + name, val, step)

        global_step = epoch * len(data_loader) + data_iter_step + 1
        step_in_epoch = data_iter_step + 1
        if data_iter_step != len(data_loader) - 1:
            if save_iter_freq > 0 and global_step % save_iter_freq == 0:
                save_model(epoch - 1, "last", step_in_epoch)
            if keep_iter_freq > 0 and global_step % keep_iter_freq == 0:
                save_model(epoch - 1, f"iter{global_step:07d}", step_in_epoch)

        save_every_steps = int(args.save_freq * len(data_loader))
        if (
            save_every_steps > 0
            and data_iter_step % save_every_steps == 0
            and data_iter_step != 0
            and data_iter_step != len(data_loader) - 1
        ):
            save_model(epoch - 1, "last", step_in_epoch)

    metric_logger.synchronize_between_processes(accelerator)
    printer.info("Averaged stats: %s", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def save_final_model(accelerator, args, epoch, model_without_ddp):
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "checkpoint-final.pth"
    to_save = {
        "args": args,
        "model": model_without_ddp.state_dict(),
        "epoch": epoch,
    }
    printer.info(">> Saving model to %s ...", checkpoint_path)
    misc.save_on_master(accelerator, to_save, checkpoint_path)


def train(args):
    ddp_static_graph = bool(getattr(args, "ddp_static_graph", True))
    ddp_find_unused_parameters = bool(getattr(args, "ddp_find_unused_parameters", False))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.accum_iter,
        mixed_precision="bf16",
        kwargs_handlers=[
            DistributedDataParallelKwargs(
                find_unused_parameters=ddp_find_unused_parameters,
                static_graph=ddp_static_graph,
            ),
            InitProcessGroupKwargs(timeout=timedelta(seconds=6000)),
        ],
    )
    device = accelerator.device

    setup_for_distributed(accelerator)
    printer.info("DDP static_graph: %s", ddp_static_graph)
    printer.info("DDP find_unused_parameters: %s", ddp_find_unused_parameters)

    printer.info("output_dir: %s", args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        dst_dir = save_current_code(outdir=args.output_dir)
        printer.info("Saving current code to %s", dst_dir)

    auto_resume_enabled = bool(getattr(args, "auto_resume", True))
    auto_resume = auto_resume_enabled and not args.resume
    if auto_resume:
        last_ckpt_fname = os.path.join(args.output_dir, "checkpoint-last.pth")
        args.resume = last_ckpt_fname if os.path.isfile(last_ckpt_fname) else None
    elif not args.resume:
        args.resume = None

    printer.info("job dir: %s", os.path.dirname(os.path.realpath(__file__)))

    seed = args.seed + accelerator.state.process_index
    printer.info("Setting seed to %s for process %s", seed, accelerator.state.process_index)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = args.benchmark

    data_loader_train = build_dataset(
        args.train_dataset,
        args.batch_size,
        args.num_workers,
        accelerator=accelerator,
        fixed_length=args.fixed_length,
    )
    _, data_loader_val = build_validation_loader(args, accelerator)

    printer.info("Loading frontend student model")
    frontend_per_layer_budget = int(getattr(args, "frontend_per_layer_budget", 8000))
    frontend_camera_budget = int(getattr(args, "frontend_camera_budget", 384))
    anchor_overflow_policy = str(getattr(args, "anchor_overflow_policy", "recent"))
    enable_track_head = int(getattr(args, "n_corres_train", 0) or 0) > 0
    if not enable_track_head:
        printer.info("Disabling track head because n_corres_train=0; this saves ~65.9M parameters.")
    model = OVGGT(
        mode=args.frontend_mode,
        frontend_pose_encoding_type=args.frontend_pose_encoding_type,
        per_layer_budget=frontend_per_layer_budget,
        camera_budget=frontend_camera_budget,
        anchor_overflow_policy=anchor_overflow_policy,
        frontend_head_checkpointing=bool(getattr(args, "frontend_head_checkpointing", False)),
        enable_track_head=enable_track_head,
    )
    printer.info(
        "Frontend budgets: per_layer_budget=%d, camera_budget=%d, anchor_overflow_policy=%s",
        frontend_per_layer_budget,
        frontend_camera_budget,
        anchor_overflow_policy,
    )
    printer.info("All model parameters: %s", sum(p.numel() for p in model.parameters()))

    printer.info("Creating train criterion = %s", args.train_criterion)
    train_criterion = eval(args.train_criterion).to(device)

    model.to(device)

    if args.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(True)
            printer.info("Enabled gradient checkpointing for OVGGT student model.")
        else:
            printer.warning(
                "gradient_checkpointing=True was requested, but OVGGT does not expose gradient_checkpointing_enable(); "
                "falling back to outer-step checkpointing only."
            )
    printer.info(
        "Frontend head checkpointing: %s",
        bool(getattr(args, "frontend_head_checkpointing", False)),
    )

    if bool(getattr(args, "activation_offload_cpu", False)):
        if hasattr(torch.autograd, "graph") and hasattr(torch.autograd.graph, "save_on_cpu"):
            printer.info("Enabled autograd activation CPU offload (save_on_cpu).")
        else:
            printer.warning(
                "activation_offload_cpu=True was requested, but torch.autograd.graph.save_on_cpu "
                "is unavailable in this runtime; ignoring."
            )

    if args.pretrained and not args.resume:
        load_student_pretrained_weights(model, args.pretrained)

    freeze_frontend_stage_a_parameters(model)
    summarize_trainable_parameters(model)

    param_groups = misc.get_parameter_groups(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler(accelerator=accelerator)

    try:
        best_so_far = misc.load_model(
            args=args,
            model_without_ddp=model,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
        )
    except Exception as exc:
        if not auto_resume:
            raise
        printer.warning(
            "Skipping auto-resume from %s after load failure: %s",
            args.resume,
            exc,
        )
        args.resume = None
        best_so_far = None
        args.start_epoch = 0
        args.start_step = 0
        if args.pretrained:
            printer.info(
                "Falling back to student pretrained weights from %s after auto-resume failure.",
                args.pretrained,
            )
            load_student_pretrained_weights(model, args.pretrained)
    if best_so_far is None:
        best_so_far = float("inf")

    accelerator.even_batches = False
    if data_loader_val is not None:
        optimizer, model, data_loader_train, data_loader_val = accelerator.prepare(
            optimizer,
            model,
            data_loader_train,
            data_loader_val,
        )
    else:
        optimizer, model, data_loader_train = accelerator.prepare(
            optimizer,
            model,
            data_loader_train,
        )

    tensorboard_log_dir = getattr(args, "logdir", args.output_dir)
    log_writer = SummaryWriter(log_dir=tensorboard_log_dir) if accelerator.is_main_process else None

    def save_model(epoch, fname, step):
        misc.save_model(
            accelerator=accelerator,
            args=args,
            model_without_ddp=accelerator.unwrap_model(model),
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
            step=step,
            fname=fname,
            best_so_far=best_so_far,
        )

    printer.info("Start frontend supervised finetuning for %s epochs", args.epochs)
    start_time = time.time()
    if data_loader_val is not None:
        printer.info(
            "Validation enabled: eval_freq=%s, eval_max_batches=%s, force_eval_on_last_epoch=%s",
            int(getattr(args, "eval_freq", 0) or 0),
            int(getattr(args, "eval_max_batches", 0) or 0),
            bool(getattr(args, "force_eval_on_last_epoch", True)),
        )
    else:
        printer.info("Validation disabled: no val_dataset/test_dataset/eval_dataset configured.")
    for epoch in range(args.start_epoch, args.epochs + 1):
        if epoch > args.start_epoch:
            should_save = (
                args.save_freq
                and np.allclose(epoch / args.save_freq, int(epoch / args.save_freq))
            ) or epoch == args.epochs
            if should_save:
                save_model(epoch - 1, "last", 0)
            if args.keep_freq and epoch % args.keep_freq == 0:
                save_model(epoch - 1, str(epoch), 0)

        if epoch >= args.epochs:
            break

        train_one_epoch(
            model=model,
            criterion=train_criterion,
            data_loader=data_loader_train,
            optimizer=optimizer,
            accelerator=accelerator,
            epoch=epoch,
            loss_scaler=loss_scaler,
            args=args,
            log_writer=log_writer,
        )

        if data_loader_val is not None and should_run_validation(epoch, args):
            val_stats = evaluate_frontend_epoch(
                model=model,
                data_loader=data_loader_val,
                accelerator=accelerator,
                epoch=epoch,
                args=args,
                loss_fn=lambda batch: supervised_loss_of_one_batch(
                    batch=batch,
                    model=model,
                    criterion=train_criterion,
                    use_amp=bool(args.amp),
                    use_activation_offload_cpu=bool(getattr(args, "activation_offload_cpu", False)),
                ),
                log_writer=log_writer,
                prefix="val",
                global_step=(epoch + 1) * len(data_loader_train),
            )
            val_loss = float(val_stats.get("loss", val_stats.get("total", float("inf"))))
            if val_loss < best_so_far:
                best_so_far = val_loss
                printer.info("New best validation loss %.6f at epoch %d", best_so_far, epoch)
                save_model(epoch, "best", 0)

    total_time = time.time() - start_time
    printer.info("Training time %s", str(datetime.timedelta(seconds=int(total_time))))
    save_final_model(accelerator, args, args.epochs, accelerator.unwrap_model(model))


@hydra.main(
    version_base=None,
    config_path=str(os.path.dirname(os.path.abspath(__file__))) + "/../config",
    config_name="finetune_frontend_blendedmvs.yaml",
)
def run(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    logdir = Path(cfg.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    try:
        train(cfg)
    except Exception:
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rank = os.environ.get("RANK", "unknown")
        local_rank = os.environ.get("LOCAL_RANK", "unknown")
        crash_path = output_dir / f"crash_rank{rank}_local{local_rank}.log"
        with open(crash_path, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        raise


if __name__ == "__main__":
    run()
