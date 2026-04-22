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

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloader_agilex import (
    AGILEX_CAMERA_NAMES,
    AGILEX_STATE_DIM,
    AgileXDataset,
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

    # Temporal windows
    parser.add_argument("--future_action_window", type=int, default=13,
                        help="Total action window (state + predictions); predicts window-1 steps")
    parser.add_argument("--past_action_window", type=int, default=0)
    parser.add_argument("--n_obs_steps", type=int, default=2,
                        help="Number of past frames fed to the vision encoder")
    parser.add_argument("--n_action_steps", type=int, default=8,
                        help="Number of predicted steps to execute per inference (receding horizon)")
    parser.add_argument("--temporal_agg", type=str, default="concat",
                        choices=["last", "mean", "concat"])

    # Model
    parser.add_argument("--model_type", type=str, default="DiT-B",
                        choices=["DiT-S", "DiT-B", "DiT-L", "DiT-XL"])
    parser.add_argument("--token_size", type=int, default=2048)
    parser.add_argument("--dropout_prob", type=float, default=0.1,
                        help="Class dropout probability for CFG")

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
    transform = build_transform(args)

    # Use explicit episode_ids so stats computation and dataset iteration agree
    # even when numeric ids include gaps (avoids lexicographic sort pitfalls
    # like episode_10 sorting before episode_2 in a naive glob slice).
    episode_ids = None
    if args.num_episodes is not None:
        episode_ids = list(range(args.num_episodes))

    stats = compute_agilex_norm_stats(
        args.data_path,
        use_robot_base=args.use_robot_base,
        episode_ids=episode_ids,
    )

    dataset = AgileXDataset(
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
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, dataset, stats


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


def train():
    args = parse_args()

    # Action dim is fully determined by the AgileX layout.
    args.action_dim = AGILEX_STATE_DIM + (2 if args.use_robot_base else 0)
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
    loader, dataset, stats = prepare_dataloader(args)
    print(f"Dataset size: {len(dataset)} | Batches: {len(loader)}")

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
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
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
        print(f"Epoch {epoch + 1} done. Avg loss={epoch_loss / len(loader):.4f}")
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
