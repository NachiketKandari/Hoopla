import argparse

from lib.augmented_generation import rag_command, summarize_command, citations_command, question_command


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    
    rag_parser = subparsers.add_parser("summarize", help="Perform Multi Document Summary")
    rag_parser.add_argument("query", type=str, help="Search query for Summarize")
    
    rag_parser = subparsers.add_parser("citations", help="Provide Citations via LLM")
    rag_parser.add_argument("query", type=str, help="Search query to provide Citations")
    
    rag_parser = subparsers.add_parser("question", help="Answer Questions via LLM")
    rag_parser.add_argument("query", type=str, help="Search query to provide answer")

    args = parser.parse_args()

    match args.command:
        case "rag":
            rag_command(args.query)
        case "summarize":
            summarize_command(args.query)
        case "citations":
            citations_command(args.query)
        case "question":
            question_command(args.query)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()