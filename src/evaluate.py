import argparse
import math

import torch
from torch.utils.data import DataLoader

from src.dataset import TokenDataset
from src.model import AstraGalaxy7


def evaluate(model, loader, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs = inputs.to(device)
            targets = targets.to(device)
            _, loss = model(inputs, targets)
            tokens = targets.numel()
            total_loss += loss.item() * tokens
            total_tokens += tokens

    mean_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(mean_loss, 20.0))
    return mean_loss, perplexity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model_config = checkpoint["model_config"]
    model = AstraGalaxy7(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    loader = DataLoader(
        TokenDataset(args.data, args.seq_len),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device == "cuda",
    )

    loss, ppl = evaluate(model, loader, device, args.max_batches)
    print(f"validation_loss: {loss:.6f}")
    print(f"perplexity: {ppl:.3f}")


if __name__ == "__main__":
    main()
