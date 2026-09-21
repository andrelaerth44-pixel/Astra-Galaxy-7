import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    mcfg = cfg["model"]
    tcfg = cfg["training"]

    set_seed(tcfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AstraGalaxy7(mcfg).to(device)
    dataset = TokenDataset(tcfg["data_path"], mcfg["max_seq_len"])
    loader = DataLoader(
        dataset,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
    )

    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    model.train()
    step = 0
    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(total=tcfg["max_steps"], desc="Astra Galaxy 7")

    while step < tcfg["max_steps"]:
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                _, loss = model(x, y)
                loss = loss / tcfg["gradient_accumulation_steps"]

            scaler.scale(loss).backward()

            if (step + 1) % tcfg["gradient_accumulation_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            step += 1
            progress.update(1)

            if step % tcfg["log_every"] == 0:
                print(f"step={step} loss={loss.item() * tcfg['gradient_accumulation_steps']:.4f}")

            if step % tcfg["save_every"] == 0:
                out = Path(tcfg["checkpoint_dir"])
                out.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), out / f"step-{step}.pt")

            if step >= tcfg["max_steps"]:
                break

    progress.close()


if __name__ == "__main__":
    main()
