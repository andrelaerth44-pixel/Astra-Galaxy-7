import argparse
import hashlib
import json
from pathlib import Path

def normalized_hash(text: str) -> str:
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Remove duplicate JSONL documents by normalized text hash.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    seen = set()
    kept = 0
    removed = 0

    with Path(args.input).open("r", encoding="utf-8") as src, Path(args.output).open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text", "")
            if not isinstance(text, str) or not text.strip():
                removed += 1
                continue
            digest = normalized_hash(text)
            if digest in seen:
                removed += 1
                continue
            seen.add(digest)
            row["text_hash"] = digest
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    print(f"kept: {kept:,}")
    print(f"removed: {removed:,}")

if __name__ == "__main__":
    main()
