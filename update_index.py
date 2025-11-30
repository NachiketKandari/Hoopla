import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from cli.lib.codebase_rag import CodebaseRAG

def main():
    print("🔄 Updating Codebase Index...")
    try:
        rag = CodebaseRAG()
        rag.build_index()
        print("✅ Index updated successfully!")
    except Exception as e:
        print(f"❌ Failed to update index: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
