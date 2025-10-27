import argparse

from lib.hybrid_search import (normalize_command,weighted_search_command,rrf_search_command)
from lib.search_utils import (DEFAULT_ALPHA_VALUE, DEFAULT_K_VALUE, DEFAULT_SEARCH_LIMIT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize the scores")
    normalize_parser.add_argument("values", metavar="N", type=float, nargs="+",help="Values to normalize")
 
    weighted_search_parser = subparsers.add_parser("weighted-search", help="Normalize the scores")
    weighted_search_parser.add_argument("query", type=str, help="Query")
    weighted_search_parser.add_argument("--alpha", type=float, default =DEFAULT_ALPHA_VALUE,help="Alpha Value")
    weighted_search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT,help="Search Limit")

    rrf_search = subparsers.add_parser("rrf-search", help="Use RRF to Search")
    rrf_search.add_argument("query", type=str, help="Query")
    rrf_search.add_argument("--k", type=float, default =DEFAULT_K_VALUE,help="K Value")
    rrf_search.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT,help="Search Limit")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            results = normalize_command(args.values)
            for score in results :
                print(f"* {score:.4f}")
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case "rrf-search" :
            rrf_search_command(args.query, args.k, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()