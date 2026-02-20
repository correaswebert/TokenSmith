import argparse
import sys
from pathlib import Path
from rich.console import Console

# Adjust path to include src if run from root
sys.path.append(".")

from src.instrumentation.logging import load_session_logs

def extract_query_data(session_id: str) -> dict[str, str]:
    """Extracts queries and their generated responses from a session log."""
    logs = load_session_logs(session_id)
    query_map = {}
    
    current_query = None
    for entry in logs:
        if entry.get("event") == "query" and "query" in entry:
            current_query = entry["query"]
        elif current_query and "generation" in entry and entry.get("event") == "query":
            # Some queries might error out, so we only map ones that successfully generated a response
            query_map[current_query] = entry["generation"].get("response_full", entry["generation"].get("response_preview", "No response logged."))
            current_query = None
            
    return query_map

def compare_sessions(base_session_id: str, new_session_id: str, query_filter: str = None):
    console = Console()
    
    base_data = extract_query_data(base_session_id)
    new_data = extract_query_data(new_session_id)
    
    if not base_data:
        console.print(f"[red]Error: Could not find query data for base session '{base_session_id}'.[/red]")
        sys.exit(1)
    if not new_data:
        console.print(f"[red]Error: Could not find query data for new session '{new_session_id}'.[/red]")
        sys.exit(1)
        
    # Find overlapping queries
    common_queries = set(base_data.keys()).intersection(set(new_data.keys()))
    
    if not common_queries:
        console.print("[yellow]No common queries found between the two sessions.[/yellow]")
        return
        
    console.print(f"[bold green]Found {len(common_queries)} common queries to compare.[/bold green]\n")
    
    for idx, query in enumerate(common_queries, 1):
        if query_filter and query_filter.lower() not in query.lower():
            continue
            
        console.print(f"[bold cyan]========================================================[/bold cyan]")
        console.print(f"[bold cyan]Query {idx}:[/bold cyan] {query}")
        console.print(f"[bold cyan]========================================================[/bold cyan]")
        
        console.print(f"\n[bold magenta]--- Base Session ({base_session_id}) ---[/bold magenta]")
        console.print(base_data[query])
        
        console.print(f"\n[bold yellow]--- New Session ({new_session_id}) ---[/bold yellow]")
        console.print(new_data[query])
        console.print("\n")

def main():
    parser = argparse.ArgumentParser(description="Comparator: Compare responses for identical queries across two sessions.")
    parser.add_argument("--base", type=str, required=True, help="The baseline session ID (e.g. 20260220_072618).")
    parser.add_argument("--new", type=str, required=True, help="The new session ID to compare against the baseline.")
    parser.add_argument("--query", type=str, help="Filter to only show a specific query containing this text.")
    args = parser.parse_args()
    
    compare_sessions(args.base, args.new, args.query)

if __name__ == "__main__":
    main()
