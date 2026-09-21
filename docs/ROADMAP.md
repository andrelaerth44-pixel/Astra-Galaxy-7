# Roadmap técnico

## Etapa 1 — fundação
- [x] Transformer decoder-only
- [x] RoPE
- [x] SwiGLU
- [x] tokenizer BPE
- [x] dataset em memória mapeada
- [x] treinamento por next-token prediction
- [x] geração sem respostas embutidas
- [ ] benchmark de perplexidade
- [ ] primeiro corpus reprodutível

## Etapa 2 — programação
- [ ] mistura de texto + código
- [ ] deduplicação
- [ ] filtros de qualidade
- [ ] splits train/validation/test
- [ ] benchmarks de Python, C/C++, JavaScript, Rust e outras linguagens
- [ ] testes de geração e correção de código

## Etapa 3 — contexto longo
- [ ] RoPE scaling validado
- [ ] atenção eficiente
- [ ] treinamento em janelas maiores
- [ ] memória/checkpointing para sequências extensas
- [ ] benchmark needle-in-a-haystack e tarefas de contexto longo

## Etapa 4 — escala
- [ ] FSDP/distributed data parallel
- [ ] checkpoints distribuídos
- [ ] gradient checkpointing
- [ ] FlashAttention quando disponível
- [ ] expansão gradual de parâmetros e tokens

## Etapa 5 — Galaxy 7 Large
Somente depois de as etapas anteriores apresentarem métricas estáveis. Qualquer meta de contexto extremo terá de ser demonstrada por benchmark, configuração e hardware publicados.
