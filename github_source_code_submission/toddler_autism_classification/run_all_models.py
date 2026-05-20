import argparse
import re
import subprocess
import sys
from pathlib import Path

from common import default_csv_path


SCRIPTS = [
    ("Logistic Regression", "logistic_regression_toddler.py"),
    ("Random Forest", "random_forest_toddler.py"),
    ("XGBoost", "xgboost_toddler.py"),
    ("CatBoost", "catboost_toddler.py"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run all toddler autism classification models.")
    parser.add_argument("--csv-path", default=str(default_csv_path()), help="Path to the toddler autism CSV dataset.")
    parser.add_argument("--output-dir", default="model_outputs", help="Directory for model stdout, stderr, and summary CSV.")
    return parser.parse_args()


def extract_accuracy(output: str):
    match = re.search(r"Accuracy:\s*([0-9]*\.?[0-9]+)", output)
    return float(match.group(1)) if match else None


def safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip()).strip("_") or "model"


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")


def save_summary_csv(path: Path, results) -> None:
    lines = ["Model,Status,Accuracy,Reason"]
    for item in results:
        accuracy = "" if item.get("accuracy") is None else f"{item['accuracy']:.6f}"
        model = (item.get("model") or "").replace('"', '""')
        status = (item.get("status") or "").replace('"', '""')
        reason = (item.get("reason") or "").replace('"', '""')
        lines.append(f'"{model}","{status}","{accuracy}","{reason}"')
    save_text(path, "\n".join(lines) + "\n")


def run_script(base_dir: Path, csv_path: Path, display_name: str, script_name: str):
    script_path = base_dir / script_name
    if not script_path.exists():
        return {
            "model": display_name,
            "status": "FAILED",
            "accuracy": None,
            "reason": "Script file does not exist.",
            "stdout": "",
            "stderr": "",
        }

    print(f"\n{'=' * 80}")
    print(f"Running model: {display_name}")
    print(f"Script: {script_name}")
    print(f"{'=' * 80}")

    result = subprocess.run(
        [sys.executable, str(script_path), "--csv-path", str(csv_path)],
        capture_output=True,
        text=True,
        cwd=str(base_dir.parent),
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("[stderr]")
        print(result.stderr)

    if result.returncode != 0:
        return {
            "model": display_name,
            "status": "FAILED",
            "accuracy": None,
            "reason": f"Exit code {result.returncode}",
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
        }

    accuracy = extract_accuracy(result.stdout)
    return {
        "model": display_name,
        "status": "OK",
        "accuracy": accuracy,
        "reason": "",
        "stdout": result.stdout or "",
        "stderr": result.stderr or "",
    }


def print_summary(results) -> None:
    print(f"\n{'#' * 80}")
    print("Model run summary")
    print(f"{'#' * 80}")
    print(f"{'Model':<25}{'Status':<10}{'Accuracy':<12}{'Reason'}")
    print("-" * 80)
    for item in results:
        accuracy = f"{item['accuracy']:.4f}" if item["accuracy"] is not None else "-"
        print(f"{item['model']:<25}{item['status']:<10}{accuracy:<12}{item['reason']}")


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = base_dir.parent / output_dir

    results = []
    for model_name, script in SCRIPTS:
        item = run_script(base_dir, csv_path, model_name, script)
        results.append(item)

        stem = safe_filename(model_name)
        save_text(output_dir / f"{stem}.stdout.txt", item.get("stdout", ""))
        save_text(output_dir / f"{stem}.stderr.txt", item.get("stderr", ""))

    print_summary(results)
    summary_csv = output_dir / "model_summary.csv"
    save_summary_csv(summary_csv, results)
    print(f"\nSaved model outputs to: {output_dir}")
    print(f"Saved summary CSV to: {summary_csv}")


if __name__ == "__main__":
    main()

