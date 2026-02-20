import argparse
import pathlib
import sys
import json
from rich.console import Console

# Adjust path to include src if run from root
sys.path.append(".")

from src.config import RAGConfig
from src.main import get_answer, load_artifacts
from src.instrumentation.logging import init_logger, get_logger, load_session_logs
from src.retriever import FAISSRetriever, BM25Retriever, IndexKeywordRetriever
from src.ranking.ranker import EnsembleRanker

def parse_args():
    parser = argparse.ArgumentParser(description="Scheduler: Replay TokenSmith queries from a past session log.")
    parser.add_argument("--session_id", type=str, required=True, help="The session ID to replay (e.g. 20260220_072618).")
    parser.add_argument("--config", default="config/config.yaml", help="Path to the new config to apply (default: config/config.yaml).")
    parser.add_argument("--index_prefix", default="textbook_index", help="Prefix for generated index files.")
    return parser.parse_args()

def extract_queries_from_session(session_id: str) -> list[str]:
    logs = load_session_logs(session_id)
    queries = []
    for entry in logs:
        if entry.get("event") == "query" and "query" in entry:
            queries.append(entry["query"])
    return queries

def main():
    args = parse_args()

    # Load new config
    config_path = pathlib.Path(args.config)
    if config_path.exists():
        cfg = RAGConfig.from_yaml(config_path)
    else:
        print(f"Error: {args.config} not found.")
        sys.exit(1)

    queries = extract_queries_from_session(args.session_id)
    if not queries:
        print(f"No queries found for session '{args.session_id}'. Check the logs/ folder.")
        sys.exit(1)

    print(f"Found {len(queries)} queries in session '{args.session_id}'. Replaying them...")
    print(f"Using config '{args.config}': Top-K={cfg.top_k}, Candidates={cfg.num_candidates}, Model={cfg.gen_model}")

    # Initialize Logger and artifacts
    init_logger(cfg)
    logger = get_logger()
    console = Console()

    print("\nInitializing pipeline artifacts...")
    try:
        artifacts_dir = cfg.get_artifacts_directory()
        faiss_index, bm25_index, chunks, sources, meta = load_artifacts(
            artifacts_dir=artifacts_dir, 
            index_prefix=args.index_prefix
        )

        retrievers = [
            FAISSRetriever(faiss_index, cfg.embed_model),
            BM25Retriever(bm25_index)
        ]
        
        if cfg.ranker_weights.get("index_keywords", 0) > 0:
            retrievers.append(
                IndexKeywordRetriever(cfg.extracted_index_path, cfg.page_to_chunk_map_path)
            )
        
        ranker = EnsembleRanker(
            ensemble_method=cfg.ensemble_method,
            weights=cfg.ranker_weights,
            rrf_k=int(cfg.rrf_k)
        )
        
        artifacts = {
            "chunks": chunks,
            "sources": sources,
            "retrievers": retrievers,
            "ranker": ranker,
            "meta": meta,
        }
    except Exception as e:
        print(f"ERROR: Failed to initialize artifacts: {e}")
        sys.exit(1)

    print("\nStarting Replay...")
    for idx, query in enumerate(queries, 1):
        print(f"\n========================================================")
        print(f"Query {idx}/{len(queries)}: [ {query} ]")
        print(f"========================================================")
        try:
            # We use an empty namespace for args parameter needed by get_answer
            dummy_args = argparse.Namespace(system_prompt_mode=None) 
            ans = get_answer(query, cfg, dummy_args, logger, console, artifacts=artifacts)
            logger.log_generation(ans, {"max_tokens": cfg.max_gen_tokens, "model_path": cfg.gen_model})
            logger.log_query_complete()
        except Exception as e:
            print(f"\nError during query execution: {e}")
            logger.log_error(e)

    print("\nReplay finished. Generating new session log.")
    print(f"View the new metrics with: python src/instrumentation/analyze_logs.py --session_id {logger.session_id}")

if __name__ == "__main__":
    main()
