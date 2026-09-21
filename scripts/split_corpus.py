import argparse
import hashlib
import json
from pathlib import Path

def bucket(text_hash: str) -> int:
    value = int(hashlib.sha256(text_hash.encode("utf-8")).hexdigest()[:8], 16)
    return value % 10000

def main():
    parser = argparse.ArgumentParser(description="Deterministic document-level train/validation split.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--validation-percent", type=float, default=2.0)
    args = parser.parse_args()

    if not 0.0 < args.validation_percent < 100.0:
        raise ValueError("--validation-percent must be between 0 and 100")

    threshold = int(args.validation_percent * 100)
    train_count = 0
    validation_count = 0

    with Path(args.input).open("r", encoding="utf-8") as src,          Path(args.train).open("w", encoding="utf-8") as train,          Path(args.validation).open("w", encoding="utf-8") as validation:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            text_hash = row.get("text_hash")
            text = row.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue
            if not isinstance(text_hash, str):
                text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

            if bucket(text_hash) < threshold:
                validation.write(json.dumps(row, ensure_ascii=False) + "\n")
                validation_count += 1
            else:
                train.write(json.dumps(row, ensure_ascii=False) + "\n")
                train_count += 1

    print(f"train: {train_count:,}")
    print(f"validation: {validation_count:,}")

if __name__ == "__main__":
    main()
