"""
Download all runs from a wandb project and store them locally.

Usage:
    python download_wandb.py --project entity/project [--output-dir wandb_data] [--tags tag1,tag2] [--name-filter pattern]

The script saves each run's history (as parquet), summary, config, and metadata (as JSON)
to `<output_dir>/<project_name>/<run_id>/`.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import wandb
from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download all runs from a wandb project."
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="Wandb project path, e.g. 'entity/project' or 'entity/project/runs'.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="wandb_data",
        help="Base directory to store downloaded data (default: wandb_data).",
    )
    parser.add_argument(
        "--tags",
        type=str,
        default=None,
        help="Comma-separated list of tags to filter runs by. Only runs matching ALL tags are kept.",
    )
    parser.add_argument(
        "--name-filter",
        type=str,
        default=None,
        help="Substring to match against run names. Only runs whose name contains this are kept.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Wandb API key. If not provided, reads from WANDB_API_KEY env var or .env file.",
    )
    return parser.parse_args()


def get_api(api_key: str | None) -> wandb.Api:
    """Create a wandb API client, using the provided key or reading from env."""
    if api_key:
        return wandb.Api(api_key=api_key)

    load_dotenv()
    env_key = os.getenv("WANDB_API_KEY")
    if env_key:
        return wandb.Api(api_key=env_key)

    # Fall back to default (may use ~/.netrc or WANDB_API_KEY already in env)
    return wandb.Api()


def filter_runs(
    runs: list,
    tags: list[str] | None,
    name_filter: str | None,
) -> list:
    """Filter runs by tags (all must match) and/or name substring."""
    if tags:
        tag_set = set(tags)
        runs = [r for r in runs if tag_set.issubset(set(r.tags or []))]

    if name_filter:
        runs = [r for r in runs if name_filter.lower() in (r.name or "").lower()]

    return runs


def sanitize_path_component(name: str) -> str:
    """Replace characters that are problematic in file paths."""
    return name.replace("/", "_").replace("\\", "_")


def download_runs(
    api: wandb.Api,
    project_path: str,
    output_dir: Path,
    tags: list[str] | None,
    name_filter: str | None,
) -> None:
    """Download all matching runs from a wandb project."""
    # Normalize project path
    # entity/project -> wandb project path
    project_path = project_path.rstrip("/")
    if project_path.endswith("/runs"):
        project_path = project_path[:-5]

    runs = api.runs(project_path)
    runs = list(runs)  # materialize

    print(f"Found {len(runs)} total runs in project '{project_path}'.")

    runs = filter_runs(runs, tags, name_filter)
    print(f"After filtering: {len(runs)} runs to download.")

    if not runs:
        print("No runs match the filters. Exiting.")
        return

    safe_project = sanitize_path_component(project_path)
    project_dir = output_dir / safe_project
    project_dir.mkdir(parents=True, exist_ok=True)

    for i, run in enumerate(runs):
        run_id = run.id
        run_name = run.name or run_id
        run_dir = project_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{i + 1}/{len(runs)}] Downloading run '{run_name}' (id={run_id}) ...")

        try:
            # History (scalar metrics logged over time)
            history_df: pd.DataFrame = run.history()
            history_path = run_dir / "history.parquet"
            history_df.to_parquet(history_path, index=False)
            print(f"    History: {len(history_df)} rows -> {history_path}")

            # Summary (final aggregated values)
            summary = dict(run.summary)
            # Convert non-serializable values to strings
            summary_clean = {}
            for k, v in summary.items():
                try:
                    json.dumps({k: v})
                    summary_clean[k] = v
                except (TypeError, ValueError):
                    summary_clean[k] = str(v)
            summary_path = run_dir / "summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary_clean, f, indent=2, default=str)
            print(f"    Summary -> {summary_path}")

            # Config (hyperparameters)
            config = dict(run.config)
            config_path = run_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2, default=str)
            print(f"    Config -> {config_path}")

            # Metadata
            metadata = {
                "id": run.id,
                "name": run.name,
                "project": project_path,
                "tags": run.tags,
                "state": run.state,
                "url": run.url,
                "created_at": str(run.created_at) if run.created_at else None,
                "notes": run.notes,
            }
            metadata_path = run_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)
            print(f"    Metadata -> {metadata_path}")

        except Exception as e:
            print(f"    ERROR downloading run '{run_id}': {e}", file=sys.stderr)
            continue

    # Write an index of all runs
    index = []
    for run in runs:
        index.append({
            "id": run.id,
            "name": run.name,
            "tags": run.tags,
            "state": run.state,
        })
    index_path = project_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2, default=str)
    print(f"\nRun index -> {index_path}")

    print(f"\nDone. Downloaded {len(runs)} runs to '{project_dir}'.")


def main() -> None:
    args = parse_args()
    api = get_api(args.api_key)

    tags_list = args.tags.split(",") if args.tags else None

    output_dir = Path(args.output_dir)

    download_runs(
        api=api,
        project_path=args.project,
        output_dir=output_dir,
        tags=tags_list,
        name_filter=args.name_filter,
    )


if __name__ == "__main__":
    main()
