from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


def train_tokenizer(input_files, output_path, vocab_size=32000):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
    )
    tokenizer.train([str(p) for p in input_files], trainer)
    tokenizer.save(str(output_path))
    return tokenizer


def encode_file(tokenizer_path, input_path, output_path):
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    text = Path(input_path).read_text(encoding="utf-8")
    ids = tokenizer.encode(text).ids
    Path(output_path).write_text(" ".join(map(str, ids)), encoding="utf-8")
    return len(ids)
