import argparse
from lib.hybrid_search import (HybridSearch)
from collections import defaultdict
from lib.search_utils import (
    DEFAULT_K_VALUE,
    DEFAULT_SEARCH_LIMIT,
    load_movies, load_testcases,
)

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    documents = load_movies()
    hybrid_search = HybridSearch(documents)

    testcases = load_testcases()
    relevant_retrieved = {}

    for testcase in testcases:
        relevant_retrieved['number'] = 0
        relevant_retrieved['retrieved'] = []
        results = hybrid_search.rrf_search(testcase['query'],DEFAULT_K_VALUE,limit)
        for result in results:
            relevant_retrieved['retrieved'].append(result['title'])
            if result['title'] in testcase['relevant_docs']:
                relevant_retrieved['number'] += 1
        precision = relevant_retrieved['number'] / limit
        recall = relevant_retrieved['number'] / len(testcase['relevant_docs'])
        f1 = 2 * (precision * recall) / (precision + recall)

        print(f"\n- Query: {testcase['query']} \n\t- Precision@{limit}: {precision:.4f} \n\t- Recall@{limit}: {recall:.4f} \n\t- F1 Score@{limit}: {f1:.4f} \n\t- Retrieved: {relevant_retrieved['retrieved']} \n\t- Relevant: {testcase['relevant_docs']}")

if __name__ == "__main__":
    main()