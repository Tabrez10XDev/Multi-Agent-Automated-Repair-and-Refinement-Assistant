import ast
import os
import chromadb
from chromadb.utils import embedding_functions

import tracing


def extract_functions(source_code: str, filename: str) -> list:
    chunks = []

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return [{
            "name": filename,
            "code": source_code,
            "docstring": "",
            "filename": filename,
            "class_name": None
        }]

    source_lines = source_code.split("\n")

    class_ranges = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_ranges[node.name] = (node.lineno, node.end_lineno)

    def get_parent_class(func_lineno: int) -> str:
        for class_name, (start, end) in class_ranges.items():
            if start <= func_lineno <= end:
                return class_name
        return None

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        start = node.lineno - 1
        end = node.end_lineno
        func_code = "\n".join(source_lines[start:end])
        docstring = ast.get_docstring(node) or ""
        parent_class = get_parent_class(node.lineno)
        qualified_name = f"{parent_class}.{node.name}" if parent_class else node.name

        chunks.append({
            "name": qualified_name,
            "code": func_code,
            "docstring": docstring,
            "filename": filename,
            "class_name": parent_class
        })

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        start = node.lineno - 1
        end = node.end_lineno
        class_code = "\n".join(source_lines[start:end])
        docstring = ast.get_docstring(node) or ""

        chunks.append({
            "name": node.name,
            "code": class_code,
            "docstring": docstring,
            "filename": filename,
            "class_name": None
        })

    return chunks


def build_index(repo_path: str, collection_name: str = "codebase") -> chromadb.Collection:
    client = chromadb.Client()

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    embedding_fn = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text"
    )

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )

    documents = []
    metadatas = []
    ids = []

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
            relative = os.path.relpath(filepath, repo_path)

            # skip everything except user source files
            if relative.startswith("agents/"):
                continue
            if filename in ("orchestrator.py", "agent.py"):
                continue
            if filename.startswith("test_"):
                continue
            if "_tests" in filename:
                continue

            with open(filepath, "r") as f:
                source = f.read()

            chunks = extract_functions(source, relative)

            for chunk in chunks:
                document = f"Function: {chunk['name']}\n{chunk['docstring']}\n{chunk['code']}"
                doc_id = f"{relative}::{chunk['name']}"

                documents.append(document)
                metadatas.append({
                    "filename": chunk["filename"],
                    "function_name": chunk["name"],
                    "class_name": chunk.get("class_name"),
                    "code": chunk["code"]
                })
                ids.append(doc_id)

    if documents:
        # Embedding work, not decode work: logged with kind="embedding" so
        # downstream analysis can exclude it from GPU decode accounting.
        with tracing.embedding_op("build_index", n_documents=len(documents)):
            collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        print(f"[Retriever] Indexed {len(documents)} functions from repo")
    else:
        print("[Retriever] Warning: no functions found to index")

    return collection


def retrieve_relevant_code(collection, issue_text: str, top_k: int = 5) -> dict:
    with tracing.embedding_op("retrieve", n_documents=1):
        results = collection.query(
            query_texts=[issue_text],
            n_results=min(top_k, collection.count())
        )

    relevant = {}

    if not results or not results["metadatas"]:
        return relevant

    for metadata in results["metadatas"][0]:
        filename = metadata["filename"]
        code = metadata["code"]

        # never return agent infrastructure files as relevant code
        # these should never be modified by the coder
        protected = [
            "orchestrator.py",
            "agent.py",
        ]
        if filename.startswith("agents/"):
            continue
        if filename in protected:
            continue

        if filename not in relevant:
            relevant[filename] = ""
        relevant[filename] += code + "\n\n"

    print(f"[Retriever] Retrieved {len(results['metadatas'][0])} functions")
    for meta in results["metadatas"][0]:
        filename = meta["filename"]
        if not filename.startswith("agents/") and filename not in ["orchestrator.py", "agent.py"]:
            print(f"  - {filename}::{meta['function_name']}")

    return relevant

def update_index(collection, repo_path: str, filename: str):
    filepath = os.path.join(repo_path, filename)

    with open(filepath, "r") as f:
        source = f.read()

    chunks = extract_functions(source, filename)

    documents = []
    metadatas = []
    ids = []

    for chunk in chunks:
        document = f"Function: {chunk['name']}\n{chunk['docstring']}\n{chunk['code']}"
        doc_id = f"{filename}::{chunk['name']}"

        documents.append(document)
        metadatas.append({
            "filename": chunk["filename"],
            "function_name": chunk["name"],
            "class_name": chunk.get("class_name"),
            "code": chunk["code"]
        })
        ids.append(doc_id)

    if documents:
        with tracing.embedding_op("update_index", n_documents=len(documents)):
            collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        print(f"[Retriever] Updated index for {filename}")