# Dados do Astra Galaxy 7

Esta pasta define o contrato de dados do pre-treinamento.

Formato de entrada: JSONL, um documento por linha:
{"text":"um documento completo aqui"}

O treinamento usa previsao do proximo token. Nao coloque respostas artificiais, listas de perguntas/respostas ou regras de dialogo apenas para ensinar comportamentos especificos.

Use somente material proprio, dominio publico ou datasets com licenca compativel.

Pipeline:
raw JSONL -> limpeza -> tokenizer -> tokens -> Transformer -> loss -> backpropagation -> pesos.

Arquivos grandes de dados e checkpoints nao devem ser commitados no Git. Registre origem, versao, licenca e hash dos datasets usados.