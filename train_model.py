import argparse
from pathlib import Path

from src.disease_risk import DiseaseRiskModel, load_cases_from_directory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train disease relative-risk model from surveillance Excel files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
        help="Directory containing .xlsx files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model_artifact.json"),
        help="Path to save trained model artifact.",
    )
    args = parser.parse_args()

    cases, summary = load_cases_from_directory(args.data_dir)
    if not cases:
        raise SystemExit("No valid training rows were found.")

    model = DiseaseRiskModel()
    model.fit(cases, training_summary=summary)
    model.save(args.output)

    print(f"Saved model to: {args.output}")
    print(f"Total cases: {summary.get('total_cases', 0)}")
    print("Disease counts:")
    for disease, count in sorted(summary.get("disease_counts", {}).items()):
        print(f"  {disease}: {count}")


if __name__ == "__main__":
    main()
