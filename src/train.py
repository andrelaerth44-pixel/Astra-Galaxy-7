import argparse
import math
import random
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import TokenDataset
from .model import AstraGalaxy7


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return max(1, step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def main():
    parser = argparse.ArgumentParser(description="Pretrain Astra Galaxy 7 with next-token prediction.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    mcfg = cfg["model"]
    tcfg = cfg["training"]

    set_seed(tcfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AstraGalaxy7(mcfg).to(device)
    dataset = TokenDataset(tcfg["data_path"], mcfg["max_seq_len"])
    if len(dataset) == 0:
        raise RuntimeError(
            f"No training sequences found in {tcfg['data_path']}. "
            "Build and tokenize a dataset first."
        )

    loader = DataLoader(
        dataset,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        drop_last=True,
        pin_memory=device == "cuda",
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
        betas=(0.9, 0.95),
    )
    scheduler = build_scheduler(
        optimizer,
        tcfg["warmup_steps"],
        tcfg["max_optimizer_steps"],
    )

    use_amp = device == "cuda"
    amp_dtype = (
        torch.bfloat16
        if use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp and amp_dtype == torch.float16,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    progress = tqdm(total=tcfg["max_optimizer_steps"], desc="Astra Galaxy 7")

    while optimizer_step < tcfg["max_optimizer_steps"]:
        for micro_step, (x, y) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                _, loss = model(x, y)
                scaled_loss = loss / tcfg["gradient_accumulation_steps"]

            scaler.scale(scaled_loss).backward()

            should_step = (
                (micro_step + 1) % tcfg["gradient_accumulation_steps"] == 0
            )
            if not should_step:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            optimizer_step += 1
            progress.update(1)

            if optimizer_step % tcfg["log_every"] == 0:
                lr = scheduler.get_last_lr()[0]
                print(
                    f"step={optimizer_step} loss={loss.item():.4f} lr={lr:.6g}"
                )

            if optimizer_step % tcfg["save_every"] == 0:
                save_checkpoint(
                    model,
                    Path(tcfg["checkpoint_dir"]) / f"step-{optimizer_step}.pt",
                )

            if optimizer_step >= tcfg["max_optimizer_steps"]:
                break

    progress.close()
    save_checkpoint(
        model,
        Path(tcfg["checkpoint_dir"]) / "final.pt",
    )
    print("training complete")


if __name__ == "__main__":
    main()
