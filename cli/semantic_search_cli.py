#!/usr/bin/env python3

import argparse
from lib.semantic_search import (embed_query_text, embed_text, search_chunked_command, verify_model, verify_embeddings, search_command, chunk_command, semantic_chunk_command, embed_chunks_command)
from lib.search_utils import (DEFAULT_SEARCH_LIMIT, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP_SIZE,DEFAULT_MAX_CHUNK_SIZE)
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

    chunk_parser = subparsers.add_parser("chunk", help= "Chunk the query")
    chunk_parser.add_argument("query", type=str, help="Search query")
    chunk_parser.add_argument("--chunk-size", type=int, default= DEFAULT_CHUNK_SIZE, help="chunk size")
    chunk_parser.add_argument("--overlap", type=int, default= DEFAULT_OVERLAP_SIZE, help="overlap size")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help= "Chunk the query")
    semantic_chunk_parser.add_argument("query", type=str, help="Search query")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default= DEFAULT_MAX_CHUNK_SIZE, help="max chunk size")
    semantic_chunk_parser.add_argument("--overlap", type=int, default= DEFAULT_OVERLAP_SIZE, help="overlap size")

    subparsers.add_parser("embed_chunks", help= "Embed the chunks")

    semantic_search_parser = subparsers.add_parser("search_chunked", help="Search Command")
    semantic_search_parser.add_argument("query", type=str, help="Search query")
    semantic_search_parser.add_argument("--limit", type=int, default= DEFAULT_SEARCH_LIMIT, help="Limit")


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
        case "chunk":
            results = chunk_command(args.query, args.chunk_size, args.overlap)
            print(f"Chunking {len(args.query)} characters")
            for i, res in enumerate(results, 1):
                print(f"{i}. {res}")
        case "semantic_chunk":
            results = semantic_chunk_command(args.query, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.query)} characters")
            for i, res in enumerate(results, 1):
                print(f"{i}. {res}")
        case "embed_chunks":
            embed_chunks_command()
        case "search_chunked":
            results = search_chunked_command(args.query, args.limit)
            for i, res in enumerate(results, 1):
                print(f"\n{i}. {res['title']} (score: {res['score']:.4f})")
                print(f"   {res['desc']}...")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()