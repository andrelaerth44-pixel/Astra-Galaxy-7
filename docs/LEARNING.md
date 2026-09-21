# Como o Astra Galaxy 7 aprende

## Não existem respostas pré-programadas

O Astra Galaxy 7 não contém uma tabela do tipo:

```text
pergunta A -> resposta A
pergunta B -> resposta B
```

Também não existe um conjunto de `if/elif` para decidir respostas.

O código do modelo define apenas uma função parametrizada. Durante o pré-treinamento, os pesos são ajustados por gradiente para minimizar a perda de previsão do próximo token.

Fluxo:

```text
documentos permitidos
        ↓
limpeza
        ↓
tokenizador BPE
        ↓
sequências de tokens
        ↓
Transformer
        ↓
next-token loss
        ↓
backpropagation
        ↓
pesos aprendidos
        ↓
geração
```

O comportamento do modelo, portanto, vem dos **dados + arquitetura + otimização + pesos**, não de respostas escritas manualmente.

## Pré-treinamento

A tarefa base é prever o próximo token:

```text
"Python é uma linguagem"
                  ↓
              próximo token
```

Para cada posição da sequência, o modelo recebe os tokens anteriores e aprende a aumentar a probabilidade do token observado no corpus.

## Programação

Para desenvolver capacidade de código, o corpus precisa conter código de boa qualidade, documentação, testes e exemplos de engenharia que possam ser usados legalmente.

Não se deve transformar cada problema em uma resposta fixa. O objetivo é que o modelo aprenda padrões de programação e gere soluções novas.

## Contexto longo

Contextos maiores exigem mudanças de arquitetura, memória e/ou atenção eficiente. A simples alteração de um número no YAML não transforma automaticamente o modelo em um sistema capaz de processar trilhões de caracteres.

O projeto vai medir cada aumento de contexto com benchmarks reproduzíveis.
