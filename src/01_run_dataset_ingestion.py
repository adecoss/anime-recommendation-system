from __future__ import annotations

import argparse
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = BASE_DIR / "notebooks" / "01_create_dataset.ipynb"
OUTPUT_CSV = BASE_DIR / "data" / "processed" / "anime_dataset.csv"


def execute_notebook(notebook_path: Path) -> None:
    try:
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
    except ImportError as exc:
        raise SystemExit(
            "Notebook execution requires nbformat and nbconvert. "
            "Install project requirements, then run this command again."
        ) from exc

    notebook = nbformat.read(notebook_path, as_version=4)
    executor = ExecutePreprocessor(timeout=None, kernel_name="python3")
    executor.preprocess(notebook, {"metadata": {"path": str(BASE_DIR)}})
    nbformat.write(notebook, notebook_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the anime catalog ingestion notebook as a reproducible command."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute the notebook. Omit this flag for a dry run.",
    )
    args = parser.parse_args()

    print(f"Ingestion notebook: {NOTEBOOK_PATH}")
    print(f"Expected processed output: {OUTPUT_CSV}")

    if not NOTEBOOK_PATH.exists():
        raise SystemExit("Dataset ingestion notebook is missing.")

    if not args.execute:
        print("Dry run only. Use --execute to run the ingestion notebook.")
        return

    execute_notebook(NOTEBOOK_PATH)
    print("Dataset ingestion notebook executed.")


if __name__ == "__main__":
    main()
