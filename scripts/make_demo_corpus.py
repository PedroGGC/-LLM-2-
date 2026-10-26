"""Generate a deterministic synthetic corpus for infrastructure smoke tests only."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/demo.txt")
    parser.add_argument("--documents", type=int, default=128)
    args = parser.parse_args()
    if args.documents < 10:
        parser.error("use at least 10 documents")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        parser.error("refusing to overwrite an existing raw corpus")
    documents = []
    for i in range(args.documents):
        text = (f"Documento sintético {i}. O laboratório registra sementes, tokens e perdas. "
                f"Na experiência {i}, o grupo mede {i + 3} amostras e repete o protocolo. "
                "A atenção causal utiliza apenas o passado. Dados de validação ficam separados. "
                "A memória do otimizador é preservada ao retomar uma execução. ")
        documents.append(text * 6)
    path.write_text("\n\n".join(documents) + "\n", encoding="utf-8", newline="\n")
    print(path.resolve())


if __name__ == "__main__":
    main()

