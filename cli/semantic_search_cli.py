#!/usr/bin/env python3

import argparse
from lib.semantic_search import (embed_query_text, embed_text, verify_model, verify_embeddings, search_command)
from lib.search_utils import (DEFAULT_SEARCH_LIMIT)
def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify Model")

    subparsers.add_parser("verify_embeddings", help="Verify Embeddings")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embed Text")
    embed_text_parser.add_argument("query", type=str, help="Search query")
    
    embed_query_parser = subparsers.add_parser("embedquery", help="Embed Query")
    embed_query_parser.add_argument("query", type=str, help="Search query")
    
    search_parser = subparsers.add_parser("search", help="Search Command")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default= DEFAULT_SEARCH_LIMIT, help="Limit")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.query)
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search_command(args.query, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()