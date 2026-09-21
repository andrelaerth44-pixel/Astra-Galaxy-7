# Pipeline de dados

A sequência recomendada para um corpus JSONL é:

1. coletar somente material cujo uso seja permitido;
2. validar os registros;
3. deduplicar;
4. dividir por documento em treino e validação;
5. construir o tokenizer usando somente a parte de treino;
6. tokenizar treino e validação separadamente;
7. treinar;
8. medir a loss no conjunto de validação.

Exemplo:

```bash
python scripts/deduplicate_jsonl.py   --input data/raw/corpus.jsonl   --output data/clean/dedup.jsonl

python scripts/split_corpus.py   --input data/clean/dedup.jsonl   --train data/clean/train.jsonl   --validation data/clean/validation.jsonl   --validation-percent 2

python scripts/prepare_data.py   --input data/clean/train.jsonl   --output data/processed/train.txt

python scripts/prepare_data.py   --input data/clean/validation.jsonl   --output data/processed/validation.txt
```

O tokenizer deve ser treinado no treino, não na validação, para reduzir vazamento de informação de avaliação.

A validação deve permanecer separada durante toda a seleção de hiperparâmetros.

Depois da tokenização:

```bash
python src/evaluate.py   --checkpoint checkpoints/final.pt   --data data/processed/validation.bin   --seq-len 2048
```

Uma loss menor é informativa apenas dentro do mesmo protocolo de dados e avaliação; ela não prova sozinha que o modelo é melhor em programação ou em outras capacidades.
