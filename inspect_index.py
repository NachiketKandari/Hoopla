import pickle
import os
import sys

# Add project root to path
sys.path.insert(0, os.getcwd())

from cli.lib.codebase_rag import CODEBASE_KEYWORD_INDEX_PATH
from cli.lib.search_utils import CACHE_DIR

print(f"Checking {CODEBASE_KEYWORD_INDEX_PATH}...")

if not os.path.exists(CODEBASE_KEYWORD_INDEX_PATH):
    print("File not found!")
    sys.exit(1)

with open(CODEBASE_KEYWORD_INDEX_PATH, "rb") as f:
    index = pickle.load(f)

print(f"Loaded index type: {type(index)}")
print(f"Docmap size: {len(index.docmap)}")
print(f"Index size: {len(index.index)}")

# Check IDs
ids = list(index.docmap.keys())
print(f"Min ID: {min(ids)}")
print(f"Max ID: {max(ids)}")

if max(ids) > 200:
    print(f"FAIL: Max ID {max(ids)} is too large! Expected ~171.")
    sys.exit(1)
else:
    print(f"SUCCESS: Max ID {max(ids)} is reasonable.")

# Check content of a doc
first_doc = index.docmap[ids[0]]
print(f"First doc content: {first_doc}")

# Check if it looks like a movie or code
if 'title' in first_doc and 'description' in first_doc:
    print("Doc has title/description.")
else:
    print("Doc missing title/description.")
