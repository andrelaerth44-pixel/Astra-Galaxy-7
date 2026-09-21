import argparse
import json
from collections import Counter
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    docs = 0
    chars = 0
    lengths = []
    languages = Counter()
    with Path(args.input).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue
            docs += 1
            chars += len(text)
            lengths.append(len(text))
            language = row.get("language")
            if isinstance(language, str) and language:
                languages[language] += 1
    print(f"documents: {docs:,}")
    print(f"characters: {chars:,}")
    if lengths:
        print(f"mean characters/document: {sum(lengths)/len(lengths):,.1f}")
        print(f"max characters/document: {max(lengths):,}")
    for language, count in languages.most_common():
        print(f"language {language}: {count:,}")

if __name__ == "__main__":
    main()
