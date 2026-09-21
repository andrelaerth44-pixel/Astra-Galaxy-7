# Astra Galaxy 7

> Um modelo de linguagem aberto, treinável e auto-hospedável, projetado para programação, raciocínio e geração de contexto extremamente longo.

## Visão

O Astra Galaxy 7 é um projeto de modelo **open-source** criado para evoluir por etapas, começando com uma base pequena e reproduzível e crescendo para arquiteturas muito maiores.

Objetivos do projeto:

- excelência em programação;
- geração e compreensão de textos muito longos;
- contexto extensível para sequências enormes;
- treinamento reproduzível;
- pesos e código publicados quando legalmente possível;
- execução local, sem depender de uma API proprietária;
- arquitetura documentada para permitir forks e auditoria.

### Sobre "trilhões de caracteres"

Um modelo não deve ser descrito como tendo trilhões de caracteres de contexto sem uma implementação e benchmark que comprovem isso. O Astra Galaxy 7 tratará contexto longo como uma **meta de engenharia mensurável**: primeiro milhares de tokens, depois milhões e, se a arquitetura, memória e hardware permitirem, escalas muito maiores.

O objetivo é construir isso de verdade, não apenas colocar um número grande no README.

## Arquitetura inicial

A primeira versão usa um Transformer decoder-only em PyTorch, com:

- RMSNorm;
- atenção causal;
- RoPE;
- MLP SwiGLU;
- pesos de embedding compartilhados com o LM head;
- checkpoint em SafeTensors;
- mixed precision quando disponível.

A configuração inicial é deliberadamente pequena para permitir testes antes de escalar.

## Estrutura

```text
Astra-Galaxy-7/
├── config/
│   └── model.yaml
├── data/
│   ├── raw/
│   └── processed/
├── scripts/
│   └── prepare_data.py
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── tokenizer.py
│   └── train.py
├── checkpoints/
├── requirements.txt
├── .gitignore
└── README.md
```

## Dados

O repositório não embute um corpus proprietário. Use apenas dados que possam ser redistribuídos legalmente, dados públicos compatíveis com suas licenças ou dados próprios.

Formato recomendado:

```json
{"text":"conteúdo do documento..."}
```

um JSON por linha.

## Primeira execução

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/prepare_data.py --input data/raw/train.jsonl --output data/processed/train.txt

python -m src.train --config config/model.yaml
```

No Windows:

```powershell
.venv\Scripts\activate
```

## Escalonamento

O projeto será desenvolvido em fases:

1. **Galaxy-7 Nano** — validar tokenizer, dataset e treinamento.
2. **Galaxy-7 Base** — aumentar largura/profundidade e contexto.
3. **Galaxy-7 Code** — corpus e avaliação focados em programação.
4. **Galaxy-7 Long** — técnicas específicas para contexto longo.
5. **Galaxy-7 Large** — escalar parâmetros, dados e hardware somente depois de validar cada estágio.

O nome "7" é o nome da família; não significa que a primeira implementação já tenha bilhões ou trilhões de parâmetros.

## Licença

O código deste repositório é destinado a uma licença permissiva. Dados e pesos podem possuir licenças diferentes conforme a origem e os termos dos datasets/modelos utilizados.

Antes de distribuir pesos treinados, registre claramente a proveniência dos dados e as licenças aplicáveis.

## Princípio de abertura

O projeto busca máxima transparência técnica: código, configuração, metodologia, avaliações e resultados devem ser públicos sempre que possível.

Abertura não significa afirmar que qualquer uso ou conteúdo é legal ou seguro. O projeto é uma infraestrutura de pesquisa e engenharia que deve ser usada de forma responsável.

## Status

**Fase 0 — fundação do treinamento.**

O próximo marco é obter uma primeira perda de treino estável em um corpus pequeno e reproduzível. Depois disso, começamos a medir programação, raciocínio e capacidade de contexto longo.
