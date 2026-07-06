# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from .rotation import quat_to_mat, mat_to_quat


ABS_POSE_ENCODING = "absT_quaR_FoV"
REL_POSE_ENCODING = "relT_quaR_FoV"
SUPPORTED_POSE_ENCODINGS = {ABS_POSE_ENCODING, REL_POSE_ENCODING}


def _validate_pose_encoding_type(pose_encoding_type: str) -> None:
    if pose_encoding_type not in SUPPORTED_POSE_ENCODINGS:
        raise NotImplementedError(
            f"Unsupported pose encoding type: {pose_encoding_type}"
        )


def _inverse_se3(matrix_4x4: torch.Tensor) -> torch.Tensor:
    rot = matrix_4x4[..., :3, :3]
    trans = matrix_4x4[..., :3, 3:]
    rot_t = rot.transpose(-1, -2)
    inv = torch.eye(
        4,
        dtype=matrix_4x4.dtype,
        device=matrix_4x4.device,
    ).expand(matrix_4x4.shape[:-2] + (4, 4)).clone()
    inv[..., :3, :3] = rot_t
    inv[..., :3, 3:] = -torch.matmul(rot_t, trans)
    return inv


def extri_intri_to_pose_encoding(
    extrinsics,
    intrinsics,
    image_size_hw=None,  # e.g., (256, 512)
    pose_encoding_type="absT_quaR_FoV",
):
    """Convert camera extrinsics and intrinsics to a compact pose encoding.

    This function transforms camera parameters into a unified pose encoding format,
    which can be used for various downstream tasks like pose prediction or representation.

    Args:
        extrinsics (torch.Tensor): Camera extrinsic parameters with shape BxSx3x4,
            where B is batch size and S is sequence length.
            In OpenCV coordinate system (x-right, y-down, z-forward), representing camera from world transformation.
            The format is [R|t] where R is a 3x3 rotation matrix and t is a 3x1 translation vector.
        intrinsics (torch.Tensor): Camera intrinsic parameters with shape BxSx3x3.
            Defined in pixels, with format:
            [[fx, 0, cx],
             [0, fy, cy],
             [0,  0,  1]]
            where fx, fy are focal lengths and (cx, cy) is the principal point
        image_size_hw (tuple): Tuple of (height, width) of the image in pixels.
            Required for computing field of view values. For example: (256, 512).
        pose_encoding_type (str): Type of pose encoding to use. Currently only
            supports "absT_quaR_FoV" (absolute translation, quaternion rotation, field of view).

    Returns:
        torch.Tensor: Encoded camera pose parameters with shape BxSx9.
            For "absT_quaR_FoV" type, the 9 dimensions are:
            - [:3] = absolute translation vector T (3D)
            - [3:7] = rotation as quaternion quat (4D)
            - [7:] = field of view (2D)
    """

    # extrinsics: BxSx3x4
    # intrinsics: BxSx3x3

    _validate_pose_encoding_type(pose_encoding_type)
    if pose_encoding_type in SUPPORTED_POSE_ENCODINGS:
        R = extrinsics[:, :, :3, :3]  # BxSx3x3
        T = extrinsics[:, :, :3, 3]  # BxSx3

        quat = mat_to_quat(R)
        # Note the order of h and w here
        H, W = image_size_hw
        fov_h = 2 * torch.atan((H / 2) / intrinsics[..., 1, 1])
        fov_w = 2 * torch.atan((W / 2) / intrinsics[..., 0, 0])
        pose_encoding = torch.cat([T, quat, fov_h[..., None], fov_w[..., None]], dim=-1).float()

    return pose_encoding


def pose_encoding_to_extri_intri(
    pose_encoding,
    image_size_hw=None,  # e.g., (256, 512)
    pose_encoding_type="absT_quaR_FoV",
    build_intrinsics=True,
):
    """Convert a pose encoding back to camera extrinsics and intrinsics.

    This function performs the inverse operation of extri_intri_to_pose_encoding,
    reconstructing the full camera parameters from the compact encoding.

    Args:
        pose_encoding (torch.Tensor): Encoded camera pose parameters with shape BxSx9,
            where B is batch size and S is sequence length.
            For "absT_quaR_FoV" type, the 9 dimensions are:
            - [:3] = absolute translation vector T (3D)
            - [3:7] = rotation as quaternion quat (4D)
            - [7:] = field of view (2D)
        image_size_hw (tuple): Tuple of (height, width) of the image in pixels.
            Required for reconstructing intrinsics from field of view values.
            For example: (256, 512).
        pose_encoding_type (str): Type of pose encoding used. Currently only
            supports "absT_quaR_FoV" (absolute translation, quaternion rotation, field of view).
        build_intrinsics (bool): Whether to reconstruct the intrinsics matrix.
            If False, only extrinsics are returned and intrinsics will be None.

    Returns:
        tuple: (extrinsics, intrinsics)
            - extrinsics (torch.Tensor): Camera extrinsic parameters with shape BxSx3x4.
              In OpenCV coordinate system (x-right, y-down, z-forward), representing camera from world
              transformation. The format is [R|t] where R is a 3x3 rotation matrix and t is
              a 3x1 translation vector.
            - intrinsics (torch.Tensor or None): Camera intrinsic parameters with shape BxSx3x3,
              or None if build_intrinsics is False. Defined in pixels, with format:
              [[fx, 0, cx],
               [0, fy, cy],
               [0,  0,  1]]
              where fx, fy are focal lengths and (cx, cy) is the principal point,
              assumed to be at the center of the image (W/2, H/2).
    """

    intrinsics = None

    _validate_pose_encoding_type(pose_encoding_type)
    if pose_encoding_type in SUPPORTED_POSE_ENCODINGS:
        T = pose_encoding[..., :3]
        quat = pose_encoding[..., 3:7]
        fov_h = pose_encoding[..., 7]
        fov_w = pose_encoding[..., 8]

        R = quat_to_mat(quat)
        extrinsics = torch.cat([R, T[..., None]], dim=-1)

        if build_intrinsics:
            H, W = image_size_hw
            fy = (H / 2.0) / torch.tan(fov_h / 2.0)
            fx = (W / 2.0) / torch.tan(fov_w / 2.0)
            intrinsics = torch.zeros(pose_encoding.shape[:2] + (3, 3), device=pose_encoding.device)
            intrinsics[..., 0, 0] = fx
            intrinsics[..., 1, 1] = fy
            intrinsics[..., 0, 2] = W / 2
            intrinsics[..., 1, 2] = H / 2
            intrinsics[..., 2, 2] = 1.0  # Set the homogeneous coordinate to 1

    return extrinsics, intrinsics


def pose_encoding_to_world_to_camera(
    pose_encoding: torch.Tensor,
    image_size_hw=None,
    pose_encoding_type: str = ABS_POSE_ENCODING,
) -> torch.Tensor:
    extrinsics, _ = pose_encoding_to_extri_intri(
        pose_encoding,
        image_size_hw=image_size_hw,
        pose_encoding_type=pose_encoding_type,
        build_intrinsics=False,
    )
    out = torch.eye(
        4,
        dtype=extrinsics.dtype,
        device=extrinsics.device,
    ).expand(extrinsics.shape[:-2] + (4, 4)).clone()
    out[..., :3, :4] = extrinsics
    return out


def pose_encoding_to_camera_to_world(
    pose_encoding: torch.Tensor,
    image_size_hw=None,
    pose_encoding_type: str = ABS_POSE_ENCODING,
) -> torch.Tensor:
    return _inverse_se3(
        pose_encoding_to_world_to_camera(
            pose_encoding,
            image_size_hw=image_size_hw,
            pose_encoding_type=pose_encoding_type,
        )
    )


def world_to_camera_to_pose_encoding(
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size_hw,
    pose_encoding_type: str = ABS_POSE_ENCODING,
) -> torch.Tensor:
    return extri_intri_to_pose_encoding(
        world_to_camera[..., :3, :4],
        intrinsics,
        image_size_hw=image_size_hw,
        pose_encoding_type=pose_encoding_type,
    )


def compose_absolute_from_relative(
    anchor_abs_pose_encoding: torch.Tensor,
    relative_pose_encoding: torch.Tensor,
    image_size_hw,
) -> torch.Tensor:
    anchor_w2c = pose_encoding_to_world_to_camera(
        anchor_abs_pose_encoding,
        image_size_hw=image_size_hw,
        pose_encoding_type=ABS_POSE_ENCODING,
    )
    relative_w2c = pose_encoding_to_world_to_camera(
        relative_pose_encoding,
        image_size_hw=image_size_hw,
        pose_encoding_type=REL_POSE_ENCODING,
    )
    _, intrinsics = pose_encoding_to_extri_intri(
        relative_pose_encoding,
        image_size_hw=image_size_hw,
        pose_encoding_type=REL_POSE_ENCODING,
        build_intrinsics=True,
    )
    current_w2c = torch.matmul(relative_w2c, anchor_w2c)
    return world_to_camera_to_pose_encoding(
        current_w2c,
        intrinsics=intrinsics,
        image_size_hw=image_size_hw,
        pose_encoding_type=ABS_POSE_ENCODING,
    )


def relative_from_absolute_pose_encoding(
    anchor_abs_pose_encoding: torch.Tensor,
    current_abs_pose_encoding: torch.Tensor,
    image_size_hw,
) -> torch.Tensor:
    anchor_w2c = pose_encoding_to_world_to_camera(
        anchor_abs_pose_encoding,
        image_size_hw=image_size_hw,
        pose_encoding_type=ABS_POSE_ENCODING,
    )
    current_w2c = pose_encoding_to_world_to_camera(
        current_abs_pose_encoding,
        image_size_hw=image_size_hw,
        pose_encoding_type=ABS_POSE_ENCODING,
    )
    _, intrinsics = pose_encoding_to_extri_intri(
        current_abs_pose_encoding,
        image_size_hw=image_size_hw,
        pose_encoding_type=ABS_POSE_ENCODING,
        build_intrinsics=True,
    )
    relative_w2c = torch.matmul(current_w2c, _inverse_se3(anchor_w2c))
    return world_to_camera_to_pose_encoding(
        relative_w2c,
        intrinsics=intrinsics,
        image_size_hw=image_size_hw,
        pose_encoding_type=REL_POSE_ENCODING,
    )
