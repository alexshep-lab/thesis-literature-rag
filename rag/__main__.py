"""Единая точка входа: python -m rag {index|query|ask|eval} ..."""
import sys

COMMANDS = {
    "index": "rag.index",
    "query": "rag.search",
    "ask": "rag.graph",
    "eval": "rag.evaluate",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("Использование: python -m rag {index|query|ask|eval} [аргументы]")
        sys.exit(1)
    import importlib

    from .tracing import setup

    setup()  # RAG_TRACE=1 — трейсы в локальный Phoenix; без флага ничего не делает
    cmd = sys.argv.pop(1)
    sys.argv[0] = f"rag {cmd}"
    importlib.import_module(COMMANDS[cmd]).main()


if __name__ == "__main__":
    main()
