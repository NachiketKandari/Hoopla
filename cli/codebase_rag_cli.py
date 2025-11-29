import argparse
import sys
from pathlib import Path

# Add the project root to sys.path to allow imports from lib
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.lib.codebase_rag import build_codebase_index_command, search_codebase_command

def main():
    parser = argparse.ArgumentParser(description="Codebase RAG CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Index command
    subparsers.add_parser("index", help="Build the codebase index")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search the codebase")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()

    if args.command == "index":
        build_codebase_index_command()
    elif args.command == "search":
        search_codebase_command(args.query, args.limit)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
