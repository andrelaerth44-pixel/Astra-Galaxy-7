# Primeiro treinamento reproduzível

Este roteiro testa o pipeline inteiro com o corpus sintético incluído no repositório. Ele não produz um modelo de qualidade; serve para validar engenharia.

## 1. Criar o corpus de teste

```bash
python scripts/make_sample_corpus.py

python scripts/deduplicate_jsonl.py \
  --input data/raw/sample.jsonl \
  --output data/clean/dedup.jsonl

python scripts/split_corpus.py \
  --input data/clean/dedup.jsonl \
  --train data/clean/train.jsonl \
  --validation data/clean/validation.jsonl \
  --validation-percent 20
```

## 2. Preparar o texto

```bash
python scripts/prepare_data.py \
  --input data/clean/train.jsonl \
  --output data/processed/train.txt

python scripts/prepare_data.py \
  --input data/clean/validation.jsonl \
  --output data/processed/validation.txt
```

A preparação preserva quebras de linha e indentação por padrão.

## 3. Treinar o tokenizer somente no treino

```bash
python scripts/build_tokenizer.py \
  --input data/processed/train.txt \
  --output data/processed/tokenizer.json \
  --vocab-size 256
```

Depois:

```bash
python scripts/tokenize_data.py \
  --tokenizer data/processed/tokenizer.json \
  --input data/processed/train.txt \
  --output data/processed/train.bin

python scripts/tokenize_data.py \
  --tokenizer data/processed/tokenizer.json \
  --input data/processed/validation.txt \
  --output data/processed/validation.bin
```

## 4. Rodar o smoke test

```bash
python -m src.train --config config/smoke.yaml
```

O treinamento deve registrar loss e, periodicamente, validation_loss e perplexidade.

## 5. Avaliar o checkpoint final

```bash
python -m src.evaluate \
  --checkpoint checkpoints/smoke/final.pt \
  --data data/processed/validation.bin \
  --seq-len 128
```

Para um corpus real, substitua o dataset sintético por material cuja licença permita o uso pretendido e mantenha o conjunto de validação separado durante o desenvolvimento.
