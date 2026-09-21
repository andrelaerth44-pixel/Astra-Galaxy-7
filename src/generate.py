import argparse
from pathlib import Path

import torch
import yaml
from tokenizers import Tokenizer

from .model import AstraGalaxy7


@torch.no_grad()
def generate(model, input_ids, max_new_tokens, temperature=0.8, top_k=50, top_p=0.95):
    model.eval()

    for _ in range(max_new_tokens):
        context = input_ids[:, -model.max_seq_len :]
        logits, _ = model(context)
        logits = logits[:, -1, :] / max(temperature, 1e-5)

        if top_k > 0:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            cutoff = values[:, -1, None]
            logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)

        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            remove = cumulative > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_indices, sorted_logits)

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat((input_ids, next_token), dim=1)

    return input_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_cfg = cfg["model"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AstraGalaxy7(model_cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    prompt_ids = tokenizer.encode(args.prompt).ids
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    output_ids = generate(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )[0].tolist()

    print(tokenizer.decode(output_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
