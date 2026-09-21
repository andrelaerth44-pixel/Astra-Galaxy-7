# Astra Galaxy 7

> Um modelo de linguagem aberto, treinável e auto-hospedável, projetado para programação e geração de contexto muito longo.

## A regra central

**O Astra Galaxy 7 não recebe respostas pré-programadas.**

Não há tabela de perguntas e respostas, árvore de diálogos ou conjunto de `if/elif` que escolha uma resposta pronta.

O código define a arquitetura e o algoritmo de treinamento. O conteúdo aprendido vem dos dados usados no treinamento e é armazenado nos pesos do modelo. Veja [docs/LEARNING.md](docs/LEARNING.md).

## Visão

Objetivos do projeto:

- programação e engenharia de software;
- geração e compreensão de textos longos;
- contexto extensível;
- treinamento reproduzível;
- código e metodologia públicos;
- execução local e auto-hospedada;
- possibilidade de estudar, modificar e fazer fork do sistema.

### Sobre trilhões de caracteres

"Trilhões de caracteres" é uma meta de pesquisa, não uma capacidade que esta primeira versão já possua.

Para chegar lá serão necessários avanços mensuráveis em tokenizer, armazenamento, atenção eficiente, memória, treinamento distribuído e avaliação. O projeto vai registrar benchmarks em vez de declarar uma capacidade sem medi-la.

## Arquitetura

A base atual usa um Transformer decoder-only em PyTorch com:

- RMSNorm;
- atenção causal via scaled dot-product attention;
- RoPE;
- MLP SwiGLU;
- embeddings compartilhados com o LM head;
- mixed precision em CUDA;
- treinamento por previsão do próximo token.

## Pipeline de aprendizagem

```text
corpus permitido
      ↓
limpeza
      ↓
tokenizador BPE aprendido do corpus
      ↓
tokens uint32 em disco
      ↓
sequências de contexto
      ↓
Transformer
      ↓
next-token loss
      ↓
backpropagation
      ↓
pesos do Astra Galaxy 7
      ↓
geração por amostragem
```

Nada nesse pipeline precisa conhecer antecipadamente as perguntas que o usuário fará.

## Estrutura

```text
Astra-Galaxy-7/
├── config/
│   └── model.yaml
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   └── LEARNING.md
├── scripts/
│   ├── prepare_data.py
│   ├── build_tokenizer.py
│   └── tokenize_data.py
├── src/
│   ├── dataset.py
│   ├── generate.py
│   ├── model.py
│   ├── tokenizer.py
│   └── train.py
├── checkpoints/
├── requirements.txt
└── README.md
```

## Dados

Formato de entrada:

```json
{"text":"conteúdo do documento..."}
```

um documento JSON por linha.

Use somente conteúdo próprio, de domínio público ou distribuível sob licença compatível. O repositório não fornece um corpus proprietário.

## Primeira execução

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/prepare_data.py \
  --input data/raw/train.jsonl \
  --output data/processed/train.txt

python scripts/build_tokenizer.py \
  --input data/processed/train.txt \
  --output data/processed/tokenizer.json

python scripts/tokenize_data.py \
  --tokenizer data/processed/tokenizer.json \
  --input data/processed/train.txt \
  --output data/processed/train.bin

python -m src.train --config config/model.yaml

python -m src.generate \
  --config config/model.yaml \
  --tokenizer data/processed/tokenizer.json \
  --checkpoint checkpoints/final.pt \
  --prompt "Escreva uma função em Python que"
```

## Escalonamento

1. **Galaxy-7 Nano** — validar dados e treinamento.
2. **Galaxy-7 Base** — ampliar parâmetros e dados.
3. **Galaxy-7 Code** — corpus e avaliações de programação.
4. **Galaxy-7 Long** — atenção e memória para contexto muito maior.
5. **Galaxy-7 Large** — treinamento distribuído em hardware de grande escala.

O nome "7" identifica a família do projeto; não afirma um número atual de parâmetros.

## Abertura

A meta é manter código, configuração, metodologia, avaliações e resultados públicos sempre que os direitos dos dados permitirem.

O projeto pode ser livre e auto-hospedável sem fingir que uma implementação pequena já resolveu escala extrema. A capacidade deve ser conquistada pelos experimentos e benchmarks.
