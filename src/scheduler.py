import argparse
import pathlib
import sys
import logging
from rich.console import Console

# Adjust path to include src if run from root
sys.path.append(".")

from src.config import RAGConfig
from src.main import get_answer, load_artifacts
from src.instrumentation.logging import init_logger, get_logger
from src.retriever import FAISSRetriever, BM25Retriever, IndexKeywordRetriever
from src.ranking.ranker import EnsembleRanker

def parse_args():
    parser = argparse.ArgumentParser(description="Scheduler: Run TokenSmith with high-spec configuration.")
    parser.add_argument("--query", type=str, required=True, help="The query to run.")
    parser.add_argument("--index_prefix", default="textbook_index", help="Prefix for generated index files.")
    parser.add_argument("--system_prompt_mode", default="detailed", help="System prompt mode (default: detailed).")
    parser.add_argument("--model_path", default="models/qwen2.5-7b-instruct-q5_k_m.gguf", help="Path to high-spec model.")
    return parser.parse_args()

def main():
    args = parse_args()

    config_path = pathlib.Path("config/config.yaml")
    if config_path.exists():
        cfg = RAGConfig.from_yaml(config_path)
    else:
        print("Error: config/config.yaml not found.")
        sys.exit(1)

    print(f"Applying High-Spec Overrides...")
    cfg.top_k = 15
    cfg.num_candidates = 80
    cfg.rrf_k = 100
    cfg.max_gen_tokens = 1024
    cfg.gen_model = args.model_path 
    cfg.system_prompt_mode = args.system_prompt_mode

    print(f"Configuration: Top-K={cfg.top_k}, Candidates={cfg.num_candidates}, Model={cfg.gen_model}")

    init_logger(cfg)
    logger = get_logger()
    console = Console()

    print("Initializing artifacts...")
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

    print(f"\nRunning Query: {args.query}\n")
    try:
        ans = get_answer(args.query, cfg, args, logger, console, artifacts=artifacts)
        logger.log_generation(ans, {"max_tokens": cfg.max_gen_tokens, "model_path": cfg.gen_model})
        logger.log_query_complete()
    except Exception as e:
        print(f"Error during generation: {e}")
        logger.log_error(e)

if __name__ == "__main__":
    main()
