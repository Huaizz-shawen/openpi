import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


class _DotSlashPathAlias:
    """Populate dot/slash key aliases so repack transforms accept either style."""

    def __call__(self, data):
        flat = transforms.flatten_dict(data)
        aliased = dict(flat)

        def _set_alias(dst: str, src: str) -> None:
            if dst not in aliased and src in aliased:
                aliased[dst] = aliased[src]

        for key, value in flat.items():
            if "/" in key:
                dot_key = key.replace("/", ".")
                aliased.setdefault(dot_key, value)
            if "." in key:
                slash_key = key.replace(".", "/")
                aliased.setdefault(slash_key, value)

        # Camera/state/task semantic aliases across common client schemas.
        _set_alias("observation.images.cam_front", "observation.images.cam_high")
        _set_alias("observation.images.cam_front", "observation/image")
        _set_alias("observation.images.cam_front", "observation.image")
        _set_alias("observation.images.cam_front", "observation/exterior_image_1_left")
        _set_alias("observation.images.cam_front", "observation.exterior_image_1_left")
        _set_alias("observation.images.cam_left", "observation.images.cam_left_wrist")
        _set_alias("observation.images.cam_left", "observation/wrist_image")
        _set_alias("observation.images.cam_left", "observation.wrist_image")
        _set_alias("observation.images.cam_left", "observation/wrist_image_left")
        _set_alias("observation.images.cam_left", "observation.wrist_image_left")
        _set_alias("observation.state", "observation/state")
        _set_alias("observation.state", "state")
        _set_alias("task", "prompt")

        # Build 7D state from DROID-style proprio if needed.
        if "observation.state" not in aliased:
            joint = aliased.get("observation/joint_position")
            if joint is None:
                joint = aliased.get("observation.joint_position")
            gripper = aliased.get("observation/gripper_position")
            if gripper is None:
                gripper = aliased.get("observation.gripper_position")
            if joint is not None and gripper is not None:
                aliased["observation.state"] = np.concatenate(
                    [np.asarray(joint).reshape(-1), np.asarray(gripper).reshape(-1)], axis=0
                )

        # Backfill slash variants for aliases we just synthesized.
        for key, value in list(aliased.items()):
            if "." in key:
                aliased.setdefault(key.replace(".", "/"), value)
            if "/" in key:
                aliased.setdefault(key.replace("/", "."), value)

        return transforms.unflatten_dict(aliased)


def _strip_actions_from_repack(group: transforms.Group) -> transforms.Group:
    """Remove action-label repack entries for inference-time request processing.

    Training repack transforms may include mappings like `actions <- action` to build labels.
    Inference requests do not include labels, so we drop only the `actions` output key here.
    """
    filtered_inputs: list[transforms.DataTransformFn] = []
    for transform in group.inputs:
        if isinstance(transform, transforms.RepackTransform):
            flat = transforms.flatten_dict(transform.structure)
            flat = {k: v for k, v in flat.items() if k != "actions"}
            transform = transforms.RepackTransform(transforms.unflatten_dict(flat))
        filtered_inputs.append(transform)
    return transforms.Group(inputs=filtered_inputs, outputs=group.outputs)


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # Prefer checkpoint-local norm stats for reproducibility, but fall back to the config-provided
        # assets source when deploying base checkpoints or stripped checkpoints without assets/.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        try:
            norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
        except FileNotFoundError:
            if data_config.norm_stats is None:
                raise
            logging.info(
                "Checkpoint norm stats missing at %s; falling back to config-loaded norm stats for asset_id=%s",
                checkpoint_dir / "assets",
                data_config.asset_id,
            )
            norm_stats = data_config.norm_stats

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    data_repack_for_infer = _strip_actions_from_repack(data_config.repack_transforms)
    merged_repack = transforms.Group(
        inputs=[*repack_transforms.inputs, _DotSlashPathAlias(), *data_repack_for_infer.inputs],
        outputs=[*data_repack_for_infer.outputs, *repack_transforms.outputs],
    )

    return _policy.Policy(
        model,
        transforms=[
            *merged_repack.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *merged_repack.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )
