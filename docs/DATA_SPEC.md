# Astra Galaxy 7 — Especificação do corpus

## Objetivo

Construir um corpus de pré-treinamento de alta qualidade para um modelo geral com forte capacidade de programação.

O corpus não deve ser uma coleção de respostas pré-programadas. O treinamento usa previsão do próximo token.

## Mistura inicial proposta

- 45% texto geral e educacional;
- 35% código-fonte e documentação de software;
- 10% matemática e ciência;
- 10% documentação técnica e referência.

Essas proporções são hipóteses e serão ajustadas pelos resultados.

## Requisitos

Registrar, quando disponível: fonte, licença, idioma, domínio, versão/data, hash e transformações aplicadas.

## Qualidade

Remover dados vazios/corrompidos, normalizar Unicode, detectar duplicatas, reduzir boilerplate repetitivo, separar treino/validação/teste e verificar licenças.

## Código

Preservar estrutura e indentação. Priorizar projetos e documentação cuja licença permita o uso planejado.

## Métricas

Acompanhar tokens, documentos, idiomas, domínios, duplicação, comprimento, proporção de código e perdas de validação.