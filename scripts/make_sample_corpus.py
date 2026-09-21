import json
from pathlib import Path

TEXTS = [
    "A funcao soma recebe dois numeros e retorna a soma deles.",
    "Um algoritmo transforma uma entrada em uma sequencia de operacoes.",
    "Uma arvore e uma estrutura de dados formada por nos conectados por arestas.",
    "Python permite definir funcoes, classes, modulos e estruturas de dados.",
    "Um compilador transforma codigo-fonte em uma representacao executavel.",
    "Uma rede neural ajusta seus parametros usando exemplos e uma funcao de perda.",
    "Transformers processam sequencias usando mecanismos de atencao.",
    "A previsao do proximo token e uma tarefa de aprendizado estatistico.",
    "Testes automatizados ajudam a detectar regressões durante o desenvolvimento.",
    "Sistemas distribuidos dividem trabalho entre multiplos processos ou maquinas.",
]

def main():
    output = Path("data/raw/sample.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for text in TEXTS:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    print(f"wrote {len(TEXTS)} synthetic documents to {output}")

if __name__ == "__main__":
    main()
