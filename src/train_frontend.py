import contextlib
import datetime
import json
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
from dust3r.datasets import get_data_loader
from dust3r.inference import sample_query_points
from ovggt.losses.frontend_distill import FrontendDistillLoss
from ovggt.models.ovggt import OVGGT
from ovggt.utils.pose_enc import REL_POSE_ENCODING
from vggt.models.vggt import VGGT

torch.backends.cuda.matmul.allow_tf32 = True
torch.multiprocessing.set_sharing_strategy("file_system")

printer = get_logger(__name__, log_level="DEBUG")


def setup_for_distributed(accelerator: Accelerator):
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        force = force or (accelerator.num_processes > 8)
        if accelerator.is_main_process or force:
            now = datetime.datetime.now().time()
            builtin_print(f"[{now}] ", end="")
            builtin_print(*args, **kwargs)

    builtins.print = print


def save_current_code(outdir: str) -> str:
    now = datetime.datetime.now()
    date_time = now.strftime("%m_%d-%H:%M:%S")
    dst_dir = os.path.join(outdir, "code", date_time)
    os.makedirs(dst_dir, exist_ok=True)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for dirname in ("src", "config"):
        src_path = os.path.join(project_root, dirname)
        if not os.path.exists(src_path):
            continue
        shutil.copytree(
            src_path,
            os.path.join(dst_dir, dirname),
            ignore=shutil.ignore_patterns(
                "*__pycache__*",
                "*.pyc",
                "*.png",
                "*.jpg",
                "*.zip",
            ),
            dirs_exist_ok=True,
        )
    return dst_dir


def resolve_state_dict(checkpoint_path: str, map_location="cpu") -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
        return checkpoint["model"]
    return checkpoint


def adapt_state_dict_for_model(model: OVGGT, state_dict: dict) -> dict:
    adapted_state_dict = dict(state_dict)
    if getattr(model, "track_head", None) is None:
        adapted_state_dict = {
            key: value for key, value in adapted_state_dict.items() if not key.startswith("track_head.")
        }
    return adapted_state_dict


def load_student_pretrained_weights(model: OVGGT, checkpoint_path: str) -> None:
    printer.info("Loading student pretrained weights from %s", checkpoint_path)
    pretrained_state = adapt_state_dict_for_model(
        model,
        resolve_state_dict(checkpoint_path, map_location="cpu"),
    )
    printer.info(model.load_state_dict(pretrained_state, strict=False))
    del pretrained_state


def freeze_frontend_stage_a_parameters(model: OVGGT) -> None:
    for _, param in model.named_parameters():
        param.requires_grad = True

    if hasattr(model.aggregator, "patch_embed"):
        for param in model.aggregator.patch_embed.parameters():
            param.requires_grad = False
    if hasattr(model.aggregator, "camera_token"):
        model.aggregator.camera_token.requires_grad = False
    if hasattr(model.aggregator, "register_token"):
        model.aggregator.register_token.requires_grad = False


def freeze_stage_a_scorer_only(model: OVGGT) -> None:
    """Stage A distillation: freeze all parameters, only train TokenScorer."""
    for _, param in model.named_parameters():
        param.requires_grad = False
    # Unfreeze scorer parameters
    if model.aggregator.token_scorers is not None:
        for param in model.aggregator.token_scorers.parameters():
            param.requires_grad = True


def summarize_trainable_parameters(model: OVGGT) -> None:
    total_params = 0
    frozen_params = 0
    for _, param in model.named_parameters():
        total_params += param.numel()
        if not param.requires_grad:
            frozen_params += param.numel()

    printer.info(
        "Frozen %s parameters out of %s total parameters. (%.2f%%)",
        f"{frozen_params:,}",
        f"{total_params:,}",
        100.0 * frozen_params / max(total_params, 1),
    )
    printer.info(
        "Trainable parameters: %s (%.2f%%)",
        f"{total_params - frozen_params:,}",
        100.0 * (total_params - frozen_params) / max(total_params, 1),
    )


def build_dataset(
    dataset,
    batch_size,
    num_workers,
    accelerator,
    fixed_length=False,
    shuffle=True,
    drop_last=True,
):
    printer.info("Building train data loader for dataset: %s", dataset)
    return get_data_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=True,
        shuffle=shuffle,
        drop_last=drop_last,
        accelerator=accelerator,
        fixed_length=fixed_length,
    )


def resolve_validation_dataset(args):
    for attr_name in ("val_dataset", "test_dataset", "eval_dataset"):
        dataset = getattr(args, attr_name, None)
        if dataset:
            return str(dataset), attr_name
    return None, None


def build_validation_loader(args, accelerator):
    dataset, dataset_attr = resolve_validation_dataset(args)
    if not dataset:
        return None, None
    printer.info("Building validation data loader from %s: %s", dataset_attr, dataset)
    loader = build_dataset(
        dataset,
        args.batch_size,
        args.num_workers,
        accelerator=accelerator,
        fixed_length=True,
        shuffle=False,
        drop_last=False,
    )
    return dataset_attr, loader


def should_run_validation(epoch: int, args) -> bool:
    finished_epochs = epoch + 1
    eval_freq = int(getattr(args, "eval_freq", 0) or 0)
    force_eval_on_last_epoch = bool(getattr(args, "force_eval_on_last_epoch", True))
    if eval_freq > 0 and finished_epochs % eval_freq == 0:
        return True
    if force_eval_on_last_epoch and finished_epochs == int(args.epochs):
        return True
    return False


@torch.no_grad()
def evaluate_frontend_epoch(
    model: torch.nn.Module,
    data_loader: Sized,
    accelerator: Accelerator,
    epoch: int,
    args,
    loss_fn,
    log_writer=None,
    prefix: str = "val",
    global_step: int | None = None,
):
    model_was_training = model.training
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = f"{prefix.capitalize()}: [{epoch}]"
    eval_max_batches = int(getattr(args, "eval_max_batches", 0) or 0)
    evaluated_batches = 0

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(0)
    if (
        hasattr(data_loader, "batch_sampler")
        and hasattr(data_loader.batch_sampler, "batch_sampler")
        and hasattr(data_loader.batch_sampler.batch_sampler, "set_epoch")
    ):
        data_loader.batch_sampler.batch_sampler.set_epoch(0)

    for batch in metric_logger.log_every(data_loader, args.print_freq, accelerator, header):
        normalize_batch_images_(batch)
        loss, loss_details = loss_fn(batch)
        loss_value = float(loss)
        if not math.isfinite(loss_value):
            raise FloatingPointError(
                f"Non-finite validation loss detected during {prefix}: "
                f"loss={loss_value}, details={loss_details}"
            )
        metric_logger.update(loss=loss_value, **loss_details)
        evaluated_batches += 1
        if eval_max_batches > 0 and evaluated_batches >= eval_max_batches:
            break

    metric_logger.synchronize_between_processes(accelerator)
    printer.info("Averaged %s stats: %s", prefix, metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats["eval_batches"] = float(evaluated_batches)

    if log_writer is not None and global_step is not None:
        for name, val in stats.items():
            if isinstance(val, dict):
                continue
            if isinstance(val, torch.Tensor):
                if val.ndim > 0:
                    continue
                val = val.item()
            log_writer.add_scalar(f"{prefix}/{name}", val, global_step)

    if model_was_training:
        model.train(True)
    return stats


def normalize_batch_images_(batch):
    if isinstance(batch, dict) and "img" in batch:
        batch["img"] = (batch["img"] + 1.0) / 2.0
        return
    if isinstance(batch, list):
        for view in batch:
            if isinstance(view, dict) and "img" in view:
                view["img"] = (view["img"] + 1.0) / 2.0


def build_query_points(batch, num_query_points: int = 64):
    if int(num_query_points or 0) <= 0:
        return None
    if not isinstance(batch, list) or not batch or "valid_mask" not in batch[0]:
        return None
    valid_mask = torch.as_tensor(batch[0]["valid_mask"])
    if valid_mask.dim() == 2:
        valid_mask = valid_mask.unsqueeze(0)
    return sample_query_points(valid_mask, M=int(num_query_points)).to(device=batch[0]["img"].device)


def infer_batch_device(batch, fallback: torch.device | None = None) -> torch.device:
    if isinstance(batch, list):
        for frame in batch:
            if not isinstance(frame, dict):
                continue
            for key in ("img", "valid_mask"):
                value = frame.get(key)
                if isinstance(value, torch.Tensor):
                    return value.device
    elif isinstance(batch, dict):
        for key in ("img", "valid_mask"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                return value.device
    return fallback or torch.device("cpu")


def get_module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def normalize_resume_step(resume_step: int, steps_per_epoch: int) -> int:
    resume_step = int(resume_step or 0)
    if resume_step < 0:
        return 0
    if steps_per_epoch <= 0:
        return 0
    if resume_step < steps_per_epoch:
        return resume_step
    # Backward compatibility: old checkpoints may store global step here.
    return resume_step % steps_per_epoch


def _sanitize_teacher_outputs(teacher_outputs):
    """Clamp inf/nan in teacher depth/pmap to prevent NaN in downstream loss."""
    for i, pred in enumerate(teacher_outputs.ress):
        for key in ("depth", "pts3d_in_other_view"):
            if key in pred:
                t = pred[key]
                if not torch.isfinite(t).all():
                    printer.warning(
                        "Teacher %s has inf/nan at frame %d, clamping to finite range.", key, i
                    )
                    pred[key] = torch.nan_to_num(t, nan=0.0, posinf=1e4, neginf=-1e4)
                    pred[key] = pred[key].clamp(-1e4, 1e4)


def frontend_loss_of_one_batch(
    batch,
    model,
    teacher,
    criterion,
    use_amp=False,
    num_query_points: int = 64,
    use_gradient_checkpointing: bool = False,
    use_activation_offload_cpu: bool = False,
    teacher_output_to_cpu: bool = False,
    teacher_weight_offload: bool = False,
    teacher_empty_cache: bool = False,
    distill_loss_weight: float = 1.0,
):
    if teacher_weight_offload and not teacher_output_to_cpu:
        raise ValueError("teacher_weight_offload=True requires teacher_output_to_cpu=True.")

    query_points = build_query_points(batch, num_query_points=num_query_points)
    autocast_enabled = bool(use_amp) and torch.cuda.is_available()
    autocast_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    autocast_device_type = "cuda" if torch.cuda.is_available() else "cpu"
    step_device = infer_batch_device(batch, fallback=get_module_device(model))

    with torch.amp.autocast(
        device_type=autocast_device_type,
        enabled=autocast_enabled,
        dtype=autocast_dtype,
    ):
        if teacher_weight_offload and get_module_device(teacher) != step_device:
            teacher.to(step_device)
        with torch.inference_mode():
            teacher_outputs = teacher.inference(
                batch,
                query_points=query_points,
                move_to_cpu=teacher_output_to_cpu,
                return_views=False,
            )
        if teacher_output_to_cpu:
            pin_cpu_tensor_tree_(teacher_outputs.ress)

        _sanitize_teacher_outputs(teacher_outputs)

        if teacher_weight_offload and get_module_device(teacher).type == "cuda":
            teacher.to(torch.device("cpu"))
            if teacher_empty_cache:
                torch.cuda.empty_cache()

        activation_offload_enabled = (
            bool(use_activation_offload_cpu)
            and torch.cuda.is_available()
            and hasattr(torch.autograd, "graph")
            and hasattr(torch.autograd.graph, "save_on_cpu")
        )
        activation_ctx = (
            torch.autograd.graph.save_on_cpu(pin_memory=True)
            if activation_offload_enabled
            else contextlib.nullcontext()
        )
        with activation_ctx:
            # Gradient checkpointing is enabled at model-module level (see train()).
            # Avoid wrapping the whole student forward with torch.utils.checkpoint here:
            # the frontend cache path mutates internal cache metadata and can break
            # checkpoint recomputation consistency in multi-GPU training.
            if isinstance(criterion, FrontendDistillLoss):
                teacher_preds = teacher_outputs.ress
                student_camera_pose_rel = []
                student_camera_pose_abs = []
                depth_terms = []
                pmap_terms = []
                student_track = []
                student_vis = []
                student_track_conf = []

                def accumulate_student_frame(frame_idx, frame_gt, student_pred):
                    student_camera_pose_rel.append(student_pred["camera_pose_rel"])
                    student_camera_pose_abs.append(student_pred["camera_pose"])
                    depth_term, pmap_term = criterion.compute_depth_and_pmap_terms(
                        frame_gt,
                        teacher_preds[frame_idx],
                        student_pred,
                    )
                    depth_terms.append(depth_term)
                    pmap_terms.append(pmap_term)
                    if "track" in student_pred and "track" in teacher_preds[frame_idx]:
                        student_track.append(student_pred["track"])
                        student_vis.append(student_pred["vis"])
                        student_track_conf.append(student_pred["track_conf"])

                student_outputs = model(
                    batch,
                    query_points=query_points,
                    frame_processor=accumulate_student_frame,
                    cache_results=False,
                    return_views=False,
                )
            else:
                student_outputs = model(batch, query_points=query_points)

            if student_outputs.keyframe_schedule is None:
                raise RuntimeError("Frontend training output is missing keyframe_schedule")

            # `views` are not consumed by FrontendDistillLoss; dropping references here
            # reduces peak memory in long-sequence training.
            teacher_outputs.views = None
            student_outputs.views = None

            with torch.amp.autocast(device_type=autocast_device_type, enabled=False):
                if isinstance(criterion, FrontendDistillLoss):
                    loss, loss_details = criterion.finalize_from_stream(
                        raw_batch_gt=batch,
                        teacher_outputs=teacher_outputs,
                        student_camera_pose_rel=student_camera_pose_rel,
                        student_camera_pose_abs=student_camera_pose_abs,
                        keyframe_schedule=student_outputs.keyframe_schedule,
                        depth_terms=depth_terms,
                        pmap_terms=pmap_terms,
                        student_track=student_track,
                        student_vis=student_vis,
                        student_track_conf=student_track_conf,
                    )
                else:
                    loss, loss_details = criterion(
                        batch,
                        teacher_outputs,
                        student_outputs,
                        student_outputs.keyframe_schedule,
                    )
                # Add TokenScorer distillation loss if present
                total_distill_loss = student_outputs.distill_loss
                if total_distill_loss is not None and distill_loss_weight > 0:
                    if torch.isfinite(total_distill_loss):
                        loss = loss + distill_loss_weight * total_distill_loss
                        loss_details["distill_loss"] = float(distill_loss_weight * total_distill_loss)
                    else:
                        loss_details["distill_loss"] = float("nan")
                    loss_details["total"] = float(loss)
    # The loss tensor already owns the autograd graph it needs. Dropping the
    # large output containers here avoids keeping extra references alive across
    # the rest of the training step, which is important for DDP memory headroom.
    del teacher_outputs
    del student_outputs
    del query_points
    return loss, loss_details


def pin_cpu_tensor_tree_(payload) -> None:
    if payload is None or not torch.cuda.is_available():
        return
    if isinstance(payload, list):
        for item in payload:
            pin_cpu_tensor_tree_(item)
        return
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, torch.Tensor) and value.device.type == "cpu" and not value.is_pinned():
                payload[key] = value.pin_memory()
        return


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


def train_one_epoch(
    model: torch.nn.Module,
    teacher: torch.nn.Module,
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

            # Log GPU memory before forward pass (every print_freq steps)
            if data_iter_step % args.print_freq == 0 and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                printer.info(
                    "[Memory] Step %d: Allocated=%.2fGB, Reserved=%.2fGB",
                    step, allocated, reserved
                )

            loss, loss_details = frontend_loss_of_one_batch(
                batch=batch,
                model=model,
                teacher=teacher,
                criterion=criterion,
                use_amp=bool(args.amp),
                num_query_points=int(getattr(args, "n_corres_train", 64) or 0),
                use_gradient_checkpointing=bool(getattr(args, "gradient_checkpointing", False)),
                use_activation_offload_cpu=bool(getattr(args, "activation_offload_cpu", False)),
                teacher_output_to_cpu=bool(getattr(args, "teacher_output_to_cpu", False)),
                teacher_weight_offload=bool(getattr(args, "teacher_weight_offload", False)),
                teacher_empty_cache=bool(getattr(args, "teacher_empty_cache", False)),
                distill_loss_weight=float(getattr(args, "distill_loss_weight", 1.0)),
            )
            loss_value = float(loss)

            if not math.isfinite(loss_value):
                printer.warning(
                    "Replacing non-finite loss with zero: loss=%s, rank=%s, step=%s, details=%s",
                    loss_value, accelerator.process_index, step, loss_details,
                )
                loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

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


def train(args):
    if int(args.batch_size) != 1:
        raise ValueError(
            f"Frontend training currently supports batch_size=1 only, got batch_size={args.batch_size}. "
            "Please set batch_size=1 in config."
        )

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
    frontend_total_budget = int(getattr(args, "frontend_total_budget", 200000))
    frontend_camera_budget = int(getattr(args, "frontend_camera_budget", 384))
    anchor_overflow_policy = str(getattr(args, "anchor_overflow_policy", "recent"))
    enable_track_head = int(getattr(args, "n_corres_train", 0) or 0) > 0
    if not enable_track_head:
        printer.info("Disabling track head because n_corres_train=0; this saves ~65.9M parameters.")
    model = OVGGT(
        mode=args.frontend_mode,
        frontend_pose_encoding_type=args.frontend_pose_encoding_type,
        total_budget=frontend_total_budget,
        camera_budget=frontend_camera_budget,
        anchor_overflow_policy=anchor_overflow_policy,
        frontend_head_checkpointing=bool(getattr(args, "frontend_head_checkpointing", False)),
        enable_track_head=enable_track_head,
        use_token_scorer=bool(getattr(args, "use_token_scorer", False)),
    )
    printer.info(
        "Frontend budgets: total_budget=%d, camera_budget=%d, anchor_overflow_policy=%s",
        frontend_total_budget,
        frontend_camera_budget,
        anchor_overflow_policy,
    )
    printer.info("All model parameters: %s", sum(p.numel() for p in model.parameters()))

    # Use OVGGT Legacy mode as Teacher instead of VGGT
    # VGGT processes all frames at once, causing OOM with 24 frames
    # OVGGT Legacy processes frames sequentially, using ~70% less memory
    teacher_total_budget = int(getattr(args, "teacher_total_budget", frontend_total_budget))
    printer.info(
        "Loading teacher model (OVGGT Legacy mode, total_budget=%d)",
        teacher_total_budget,
    )
    teacher = OVGGT(
        mode="legacy",
        total_budget=teacher_total_budget,
        enable_track_head=enable_track_head,
    )

    printer.info("Creating train criterion = %s", args.train_criterion)
    train_criterion = eval(args.train_criterion).to(device)

    model.to(device)
    teacher_output_to_cpu = bool(getattr(args, "teacher_output_to_cpu", False))
    teacher_weight_offload = bool(getattr(args, "teacher_weight_offload", False))
    if teacher_weight_offload and not teacher_output_to_cpu:
        raise ValueError("teacher_weight_offload=True requires teacher_output_to_cpu=True.")
    if not teacher_weight_offload:
        teacher.to(device)
    if bool(getattr(args, "teacher_empty_cache", False)):
        printer.info("Teacher CUDA allocator cache will be explicitly released after offload.")

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
    if teacher_output_to_cpu:
        printer.info("Teacher targets are offloaded to CPU between teacher and student passes.")
    if teacher_weight_offload:
        printer.info("Teacher weights stay on CPU between steps to free student/backward memory.")

    if args.pretrained and not args.resume:
        load_student_pretrained_weights(model, args.pretrained)

    teacher_path = args.teacher or args.pretrained
    if not teacher_path:
        raise ValueError(
            "Teacher checkpoint path is required for frontend distillation training. "
            "Please set `teacher` or `pretrained` in the config."
        )
    printer.info("Loading teacher weights from %s", teacher_path)
    teacher_state = adapt_state_dict_for_model(
        teacher,
        resolve_state_dict(teacher_path, map_location="cpu"),
    )
    teacher.load_state_dict(teacher_state, strict=True)
    del teacher_state

    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()

    if bool(getattr(args, "use_token_scorer", False)) and not bool(getattr(args, "finetune_full_model", False)):
        freeze_stage_a_scorer_only(model)
    else:
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
            optimizer, model, data_loader_train, data_loader_val
        )
    else:
        optimizer, model, data_loader_train = accelerator.prepare(
            optimizer, model, data_loader_train
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

    printer.info("Start frontend training for %s epochs", args.epochs)
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
            teacher=teacher,
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
                loss_fn=lambda batch: frontend_loss_of_one_batch(
                    batch=batch,
                    model=model,
                    teacher=teacher,
                    criterion=train_criterion,
                    use_amp=bool(args.amp),
                    num_query_points=int(getattr(args, "n_corres_train", 64) or 0),
                    use_gradient_checkpointing=bool(getattr(args, "gradient_checkpointing", False)),
                    use_activation_offload_cpu=bool(getattr(args, "activation_offload_cpu", False)),
                    teacher_output_to_cpu=bool(getattr(args, "teacher_output_to_cpu", False)),
                    teacher_weight_offload=bool(getattr(args, "teacher_weight_offload", False)),
                    teacher_empty_cache=bool(getattr(args, "teacher_empty_cache", False)),
                    distill_loss_weight=float(getattr(args, "distill_loss_weight", 1.0)),
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
    config_name="train_frontend_blendedmvs.yaml",
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
