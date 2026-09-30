import os
import ast
from agents.paths import is_agent_visible
from agents.retriever import build_index, retrieve_relevant_code


def get_file_structure(repo_path: str) -> dict:
    """
    Walk the repo and extract function/class signatures from Python files.
    Returns { filename: [list of function names] }
    """
    structure = {}

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".")
            and d not in ("venv", "__pycache__", "node_modules")
        ]

        for filename in files:
            if not filename.endswith(".py"):
                continue

            filepath = os.path.join(root, filename)
            relative_path = os.path.relpath(filepath, repo_path)

            try:
                with open(filepath, "r") as f:
                    source = f.read()
                tree = ast.parse(source)
                functions = [
                    node.name for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                ]
                structure[relative_path] = functions
            except Exception:
                structure[relative_path] = []

    return structure


def get_test_file_contents(repo_path: str) -> dict:
    test_files = {}
    for filename in os.listdir(repo_path):
        # catch both test_*.py and *_tests.py naming conventions
        if filename.endswith(".py") and (
            filename.startswith("test_") or filename.endswith("_tests.py")
        ):
            filepath = os.path.join(repo_path, filename)
            with open(filepath, "r") as f:
                test_files[filename] = f.read()
    return test_files

def navigate(repo_path: str, issue_text: str, affected_files: list, use_rag: bool = True) -> dict:
    """
    Main entry point for the navigator agent.

    Phase 3 upgrade:
    - Uses semantic retrieval to find relevant functions
    - Also reads test files so the coder knows what to satisfy
    - Falls back to affected_files if retrieval finds nothing
    """
    print(f"\n[Navigator] Scanning repo: {repo_path}")

    structure = get_file_structure(repo_path)
    print(f"[Navigator] Found files: {list(structure.keys())}")

    if use_rag:
        print("[Navigator] Building semantic index...")
        collection = build_index(repo_path)
        relevant_code = retrieve_relevant_code(collection, issue_text, top_k=5)
        relevant_code = {
            k: v for k, v in relevant_code.items() if is_agent_visible(k)
        }
        if not relevant_code:
            print("[Navigator] Retrieval returned nothing, falling back to affected_files")
            for filename in affected_files:
                filepath = os.path.join(repo_path, filename)
                if os.path.exists(filepath):
                    with open(filepath, "r") as f:
                        relevant_code[filename] = f.read()
    else:
        print("[Navigator] RAG disabled (ablation: hetero_no_rag), using affected_files directly.")
        collection = None
        relevant_code = {}
        for filename in affected_files:
            filepath = os.path.join(repo_path, filename)
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    relevant_code[filename] = f.read()

    # always include test files
    test_files = get_test_file_contents(repo_path)
    print(f"[Navigator] Loaded test files: {list(test_files.keys())}")

    return {
        "structure": structure,
        "relevant_code": relevant_code,
        "test_files": test_files,
        "collection": collection,
        "chunks_retrieved": len(relevant_code) if use_rag else 0
    }
    
    
    