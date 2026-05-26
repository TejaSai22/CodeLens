"""
Retrieval evaluation harness.

Indexes a repository into a throwaway vector store, runs a golden set of
questions through both dense-only and hybrid retrieval, and reports recall@k,
MRR and hit@k for each mode so retrieval changes can be judged by numbers.

Usage:
    uv run python -m evals.run_eval                 # evaluate this repo's src/
    uv run python -m evals.run_eval --repo PATH --k 5
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.ingestion.repo_loader import RepoLoader          # noqa: E402
from src.ingestion.chunker import CodeChunker             # noqa: E402
from src.ingestion.embedder import Embedder               # noqa: E402
from src.retrieval.vector_store import VectorStore        # noqa: E402
from evals.metrics import recall_at_k, reciprocal_rank, hit_at_k, first_relevant_rank  # noqa: E402


def _norm(path: str) -> str:
    return path.replace("\\", "/")


def index_repo(repo_path: Path, persist_dir: str, use_ast: bool = True):
    loader = RepoLoader(repo_path=str(repo_path))
    files = loader.get_files()
    chunker = CodeChunker(use_ast=use_ast)
    embedder = Embedder()

    chunks = []
    for file_data in files:
        chunks.extend(chunker.chunk_file(file_data))

    embeddings = embedder.embed([c.content for c in chunks])
    store = VectorStore(collection_name="eval", persist_dir=persist_dir)
    store.clear()
    store.add_chunks(chunks, embeddings)
    return store, embedder, len(files), len(chunks)


def ranked_files(store, embedder, mode: str, query: str, k: int):
    emb = embedder.embed_single(query)
    if mode == "dense":
        results = store.query(emb, n_results=k)
    else:
        results = store.hybrid_query(query_text=query, query_embedding=emb, n_results=k)
    return [_norm(r["metadata"].get("file_path", "")) for r in results]


def main():
    parser = argparse.ArgumentParser(description="CodeLens retrieval eval")
    parser.add_argument("--repo", default=str(REPO_ROOT / "src"))
    parser.add_argument("--golden", default=str(REPO_ROOT / "evals" / "golden.json"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--no-ast", action="store_true",
                        help="Force sliding-window chunking (to A/B against AST chunking)")
    args = parser.parse_args()

    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    persist_dir = tempfile.mkdtemp(prefix="codelens_eval_")

    try:
        store, embedder, n_files, n_chunks = index_repo(Path(args.repo), persist_dir, use_ast=not args.no_ast)
        mode_label = "sliding-window" if args.no_ast else "AST"
        print(f"Indexed {n_files} files / {n_chunks} chunks from {args.repo} ({mode_label} chunking)\n")

        modes = ("dense", "hybrid")
        agg = {m: {"recall": 0.0, "mrr": 0.0, "hit": 0.0} for m in modes}

        head = f"{'Question':<50} | {'expected':<30} | dense | hybrid"
        print(head)
        print("-" * len(head))

        for item in golden:
            query = item["question"]
            expected = {_norm(f) for f in item["expected_files"]}
            ranks = {}
            for m in modes:
                files = ranked_files(store, embedder, m, query, args.k)
                agg[m]["recall"] += recall_at_k(expected, files, args.k)
                agg[m]["mrr"] += reciprocal_rank(expected, files)
                agg[m]["hit"] += hit_at_k(expected, files, args.k)
                ranks[m] = first_relevant_rank(expected, files)

            exp_str = ",".join(sorted(expected))
            d = ranks["dense"] or "-"
            h = ranks["hybrid"] or "-"
            print(f"{query[:48]:<50} | {exp_str[:28]:<30} | {str(d):^5} | {str(h):^6}")

        n = len(golden)
        print(f"\n=== Summary (n={n}, k={args.k}) - rank shown is first relevant hit ===")
        print(f"{'mode':<8} | recall@k |  MRR  | hit@k")
        print("-" * 38)
        for m in modes:
            print(f"{m:<8} | {agg[m]['recall']/n:8.3f} | {agg[m]['mrr']/n:5.3f} | {agg[m]['hit']/n:5.3f}")
    finally:
        shutil.rmtree(persist_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
