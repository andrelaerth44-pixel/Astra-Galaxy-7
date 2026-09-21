# Plano de treinamento

## Fase A — smoke test

Provar que o pipeline produz perda finita e checkpoints carregáveis.

## Fase B — corpus pequeno

Confirmar que a perda de validação diminui em dados reais e medir train loss, validation loss, perplexidade, tokens/segundo, memória e estabilidade.

## Fase C — programação

Adicionar código legalmente utilizável e avaliar geração por testes executáveis em ambiente isolado.

## Fase D — escala

Aumentar parâmetros e tokens, ampliar contexto, usar treinamento distribuído e otimizar memória/atenção.

## Fase E — contexto extremo

Medir memória, latência, throughput, degradação com comprimento e recuperação de informação distante. A meta de trilhões de caracteres só será considerada alcançada mediante demonstração reproduzível.