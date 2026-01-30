import sys
print(f"Python Executable: {sys.executable}")
print(f"Path: {sys.path}")

try:
    import sentence_transformers
    print("sentence_transformers: OK")
except ImportError as e:
    print(f"sentence_transformers: FAIL {e}")

try:
    import langchain_community.embeddings
    print("langchain_community: OK")
except ImportError as e:
    print(f"langchain_community: FAIL {e}")
