import argparse
import json
from pathlib import Path

from src.disease_risk import DiseaseRiskModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a disease risk prediction.")
    parser.add_argument("--model", type=Path, default=Path("model_artifact.json"))
    parser.add_argument("--lat", type=float, required=True, help="Latitude")
    parser.add_argument("--lon", type=float, required=True, help="Longitude")
    parser.add_argument("--age", type=float, default=None, help="Age in years")
    parser.add_argument("--gender", type=str, default="unknown", help="male/female/other")
    args = parser.parse_args()

    model = DiseaseRiskModel.load(args.model)
    result = model.predict(
        latitude=args.lat,
        longitude=args.lon,
        age=args.age,
        gender=args.gender,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
