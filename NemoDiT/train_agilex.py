"""Training entry point for the AgileX / Songling real-robot adaptation.

大部分训练逻辑 (优化器、warmup、AMP、EMA、checkpoint) 与 ``train.py`` 保持一致，
只是将数据加载器替换为 ``dataloader_agilex.AgileXDataset`` 并把归一化统计一同
保存进 checkpoint，供 ``inference_agilex.py`` 在真机上反归一化使用。

典型用法::

    python train_agilex.py \
        --data_path /path/to/agilex/episodes \
        --checkpoint_dir checkpoints/agilex-pick-50 \
        --num_cameras 3 \
        --batch_size 32 \
        --epochs 500
"""

import argparse
import math
import os
import pickle
from pathlib import Path
from typing import List

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloader_agilex import (
    AGILEX_CAMERA_NAMES,
    AGILEX_STATE_DIM,
    AgileXDataset,
    agilex_action_dim,
    compute_agilex_norm_stats,
)
from model.action_model.action_model import ActionModel
from utils.ema_model import EMAModel
from utils.wandb_utils import WandbLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NemoDiT on AgileX real-robot data")

    # Data
    parser.add_argument("--data_path", type=str, required=True,
                        help="Directory containing episode_*.hdf5 (AgileX format)")
    parser.add_argument("--num_episodes", type=int, default=None,
                        help="Limit to first N episodes (default: all)")
    parser.add_argument("--num_cameras", type=int, default=3,
                        help="Number of camera views to use (default 3: cam_high, left_wrist, right_wrist)")
    parser.add_argument("--camera_names", nargs="+", default=None,
                        help=f"Camera names in HDF5 (default: {AGILEX_CAMERA_NAMES})")
    parser.add_argument("--use_robot_base", action="store_true", default=False,
                        help="Concatenate /base_action (2-D) onto qpos/action")
    parser.add_argument("--arm_delay_time", type=int, default=0,
                        help="Forward shift of action target in frames (default: 0)")
    parser.add_argument("--exclude_terminal_padding", action="store_true", default=False,
                        help="Drop episode-tail timesteps that need right-padding "
                             "(avoids 'stay still' bias at episode end)")
    parser.add_argument("--val_ratio", type=float, default=0.0,
                        help="Fraction of episodes held out for validation (default: 0.0 = no val).")
    parser.add_argument("--val_seed", type=int, default=0,
                        help="Seed for the train/val episode shuffle (default: 0).")
    parser.add_argument("--val_every", type=int, default=10,
                        help="Run validation every N epochs (only when --val_ratio > 0).")

    # Temporal windows
    parser.add_argument("--future_action_window", type=int, default=10,
                        help="Total action window (= state slot + predicted frames). The model "
                             "predicts (future_action_window - 1) frames; default 10 -> 9 predicted "
                             "(~300ms horizon @ 30Hz capture rate).")
    parser.add_argument("--past_action_window", type=int, default=0)
    parser.add_argument("--n_obs_steps", type=int, default=3,
                        help="Number of past frames fed to the vision encoder. Default 3 "
                             "gives ~67ms temporal window @ 30Hz capture, which helps the "
                             "encoder pick up motion / velocity cues vs a single frame.")
    parser.add_argument("--n_action_steps", type=int, default=4,
                        help="Number of predicted steps to execute per inference (receding horizon). "
                             "Default 4 -> re-plan every 4/publish_rate seconds (133ms @ 30Hz).")
    parser.add_argument("--temporal_agg", type=str, default="concat",
                        choices=["last", "mean", "concat"])

    # Model
    parser.add_argument("--model_type", type=str, default="DiT-B",
                        choices=["DiT-S", "DiT-B", "DiT-L", "DiT-XL"])
    parser.add_argument("--token_size", type=int, default=2048)
    parser.add_argument("--dropout_prob", type=float, default=0.0,
                        help="Class dropout probability for classifier-free guidance. "
                             "Set to 0 when you don't plan to use --cfg_scale > 1 at inference "
                             "(default). Set to 0.1 only if you want CFG.")

    # Vision
    parser.add_argument("--vision_backbone", type=str, default="resnet50",
                        choices=["resnet18", "resnet34", "resnet50", "vit_b_16", "vit_b_32"])
    parser.add_argument("--vision_pretrained", action="store_true", default=True)
    parser.add_argument("--freeze_vision", action="store_true", default=False)
    parser.add_argument("--adapter_type", type=str, default="mlp",
                        choices=["linear", "mlp", "attention_pooling"])
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--no_resize", action="store_true", default=False,
                        help="Skip image resize/crop and use the raw 480x640 frame")

    # Flow matching
    parser.add_argument("--time_sampling", type=str, default="logit_normal",
                        choices=["logit_normal", "beta", "uniform"])
    parser.add_argument("--logit_normal_loc", type=float, default=0.0)
    parser.add_argument("--logit_normal_scale", type=float, default=1.0)
    parser.add_argument("--beta_alpha", type=float, default=1.5)
    parser.add_argument("--beta_beta", type=float, default=1.0)
    parser.add_argument("--num_timestep_buckets", type=int, default=1000)
    parser.add_argument("--num_inference_steps", type=int, default=10)

    # Optimization
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--warmup_type", type=str, default="linear",
                        choices=["linear", "cosine"])

    # EMA
    parser.add_argument("--use_ema", action="store_true", default=False)
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0)
    parser.add_argument("--ema_power", type=float, default=0.6667)
    parser.add_argument("--ema_max_value", type=float, default=0.9999)

    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/agilex")
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--stats_name", type=str, default="dataset_stats.pkl")

    # Device / logging
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="nemodit_agilex")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)

    return parser.parse_args()


def build_transform(args: argparse.Namespace) -> transforms.Compose:
    if args.no_resize:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def prepare_dataloader(args: argparse.Namespace):
    """Build train (and optional val) dataloaders + normalization stats.

    Stats are computed **only on the training split** and shared with the val
    set, mirroring agx_robot ACT's train/val convention.
    """
    transform = build_transform(args)

    # Use explicit episode_ids so stats computation and dataset iteration agree
    # even when numeric ids include gaps (avoids lexicographic sort pitfalls
    # like episode_10 sorting before episode_2 in a naive glob slice).
    if args.num_episodes is not None:
        all_ids = list(range(args.num_episodes))
    else:
        # Fall back to scanning the directory.
        from dataloader_agilex import _list_episode_files  # local import
        all_ids = []
        for path in _list_episode_files(args.data_path, None):
            stem = Path(path).stem
            try:
                all_ids.append(int(stem.split("_")[-1]))
            except ValueError:
                continue
        all_ids.sort()

    # Train / val split.
    val_ids: List[int] = []
    train_ids = list(all_ids)
    if args.val_ratio > 0.0 and len(all_ids) > 1:
        import random

        rng = random.Random(args.val_seed)
        shuffled = list(all_ids)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(args.val_ratio * len(shuffled))))
        val_ids = sorted(shuffled[:n_val])
        train_ids = sorted(shuffled[n_val:])
    print(f"[Split] train_episodes={len(train_ids)} val_episodes={len(val_ids)}")

    stats = compute_agilex_norm_stats(
        args.data_path,
        use_robot_base=args.use_robot_base,
        episode_ids=train_ids,
    )

    def _make_dataset(episode_ids: List[int]) -> AgileXDataset:
        return AgileXDataset(
            data_path=args.data_path,
            norm_stats=stats,
            future_action_window=args.future_action_window,
            past_action_window=args.past_action_window,
            transform=transform,
            camera_names=args.camera_names,
            num_cameras=args.num_cameras,
            n_obs_steps=args.n_obs_steps,
            use_robot_base=args.use_robot_base,
            arm_delay_time=args.arm_delay_time,
            episode_ids=episode_ids,
            exclude_terminal_padding=args.exclude_terminal_padding,
        )

    train_dataset = _make_dataset(train_ids)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    val_dataset = None
    if val_ids:
        val_dataset = _make_dataset(val_ids)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    return train_loader, train_dataset, val_loader, val_dataset, stats


def build_model(args: argparse.Namespace) -> ActionModel:
    return ActionModel(
        token_size=args.token_size,
        model_type=args.model_type,
        in_channels=args.action_dim,
        future_action_window_size=args.future_action_window,
        past_action_window_size=args.past_action_window,
        time_sampling=args.time_sampling,
        logit_normal_loc=args.logit_normal_loc,
        logit_normal_scale=args.logit_normal_scale,
        beta_alpha=args.beta_alpha,
        beta_beta=args.beta_beta,
        num_timestep_buckets=args.num_timestep_buckets,
        use_vision_condition=True,
        vision_backbone_type=args.vision_backbone,
        vision_pretrained=args.vision_pretrained,
        num_cameras=args.num_cameras,
        freeze_vision_backbone=args.freeze_vision,
        adapter_type=args.adapter_type,
        class_dropout_prob=args.dropout_prob,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        temporal_agg=args.temporal_agg,
    )


def get_scheduler(optimizer, args: argparse.Namespace):
    if args.warmup_epochs <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def lr_lambda(current_epoch: int) -> float:
        if current_epoch < args.warmup_epochs:
            if args.warmup_type == "linear":
                return (current_epoch + 1) / args.warmup_epochs
            return 0.5 * (1 - math.cos(math.pi * (current_epoch + 1) / args.warmup_epochs))
        progress = (current_epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, scheduler, scaler, ema_model, stats, epoch,
                    global_step, args, filename=None):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if filename is None:
        filename = f"{epoch}.pt"
    path = os.path.join(args.checkpoint_dir, filename)

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "args": vars(args),
        "norm_stats": {k: v.tolist() for k, v in stats.items()},
    }
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if ema_model is not None:
        payload["ema_state_dict"] = ema_model.state_dict()

    torch.save(payload, path)
    torch.save(payload, os.path.join(args.checkpoint_dir, "latest.pt"))
    print(f"[Checkpoint] Saved to {path}")


def load_checkpoint(model, optimizer, scheduler, scaler, ema_model, path):
    print(f"[Checkpoint] Resuming from {path}")
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    if ema_model is not None and "ema_state_dict" in ckpt:
        ema_model.load_state_dict(ckpt["ema_state_dict"])
    return ckpt["epoch"], ckpt["global_step"]


@torch.no_grad()
def run_validation(model, val_loader, device, use_amp: bool) -> float:
    """Compute mean Flow-Matching loss over the validation set."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in val_loader:
        images = batch["images"].to(device)
        state = batch["state"].to(device)
        actions = batch["actions"].to(device)
        if use_amp:
            with autocast():
                loss = model.loss(x=actions, images=images, state=state)
        else:
            loss = model.loss(x=actions, images=images, state=state)
        total_loss += loss.item()
        n_batches += 1
    model.train()
    return total_loss / max(1, n_batches)


def train():
    args = parse_args()

    # AgileX is always dual-arm 14-D (+2 if robot base is enabled).
    args.action_dim = agilex_action_dim(args.use_robot_base)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"action_dim={args.action_dim} (AgileX dual-arm{' + base' if args.use_robot_base else ''})")
    print(f"n_obs_steps={args.n_obs_steps} n_action_steps={args.n_action_steps} "
          f"future_action_window={args.future_action_window} temporal_agg={args.temporal_agg}")

    wandb_logger = None
    if args.use_wandb:
        run_name = args.wandb_name or f"{Path(args.checkpoint_dir).name}_{args.model_type}"
        wandb_logger = WandbLogger(
            project_name=args.wandb_project,
            run_name=run_name,
            config=vars(args),
            entity=args.wandb_entity,
        )

    print("Loading dataset and computing normalization stats...")
    train_loader, train_dataset, val_loader, val_dataset, stats = prepare_dataloader(args)
    print(f"Train dataset size: {len(train_dataset)} | Batches: {len(train_loader)}")
    if val_loader is not None:
        print(f"Val   dataset size: {len(val_dataset)} | Batches: {len(val_loader)}")

    # Persist normalization stats standalone for convenience (inference also reads from ckpt).
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(os.path.join(args.checkpoint_dir, args.stats_name), "wb") as f:
        pickle.dump(stats, f)
    print(f"[Checkpoint] Normalization stats saved to {args.checkpoint_dir}/{args.stats_name}")

    model = build_model(args).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: total={total:,} trainable={trainable:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    scheduler = get_scheduler(optimizer, args)
    scaler = GradScaler() if args.use_amp else None

    ema_model = None
    if args.use_ema:
        ema_model = EMAModel(
            model.net,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            max_value=args.ema_max_value,
        )

    start_epoch, global_step = 0, 0
    if args.resume:
        start_epoch, global_step = load_checkpoint(model, optimizer, scheduler, scaler, ema_model, args.resume)

    print("Starting training...")
    model.train()
    best_val_loss = float("inf")
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            images = batch["images"].to(device)
            state = batch["state"].to(device)
            actions = batch["actions"].to(device)

            optimizer.zero_grad()
            if args.use_amp:
                with autocast():
                    loss = model.loss(x=actions, images=images, state=state)
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model.loss(x=actions, images=images, state=state)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            if ema_model is not None:
                ema_model.step(model.net)

            epoch_loss += loss.item()
            global_step += 1
            if wandb_logger:
                log_dict = {
                    "train/loss": loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/epoch": epoch + 1,
                    "train/global_step": global_step,
                }
                if ema_model is not None:
                    log_dict["train/ema_decay"] = ema_model.decay
                wandb_logger.log(log_dict, step=global_step)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg": f"{epoch_loss / (batch_idx + 1):.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

        scheduler.step()
        train_avg = epoch_loss / max(1, len(train_loader))
        print(f"Epoch {epoch + 1} done. train_loss={train_avg:.4f}")

        # Validation pass.
        if val_loader is not None and args.val_every > 0 \
                and ((epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1):
            val_loss = run_validation(model, val_loader, device, args.use_amp)
            print(f"           val_loss  ={val_loss:.4f}")
            if wandb_logger:
                wandb_logger.log({"val/loss": val_loss, "val/epoch": epoch + 1},
                                 step=global_step)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, scheduler, scaler, ema_model, stats,
                                epoch + 1, global_step, args, filename="best.pt")
                print(f"           best val so far -> saved best.pt (loss={val_loss:.4f})")

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, scaler, ema_model, stats,
                            epoch + 1, global_step, args)

    print("Training complete.")
    save_checkpoint(model, optimizer, scheduler, scaler, ema_model, stats,
                    args.epochs, global_step, args, filename="final.pt")
    if wandb_logger:
        wandb_logger.finish()


if __name__ == "__main__":
    train()
