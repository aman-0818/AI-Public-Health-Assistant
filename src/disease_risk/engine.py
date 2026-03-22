"""
src/disease_risk/engine.py
--------------------------
Core prediction engine for the Disease Risk Sentinel app.

HOW IT WORKS (overview):
  1. load_cases_from_directory() reads all .xlsx files from the dataset folder.
     It parses each row, extracts lat/lon/age/gender/disease, and returns a list of cases.

  2. DiseaseRiskModel.fit() takes those cases and:
       - Builds a spatial grid index (cells of ~0.25 degrees) for fast neighbor lookup
       - Computes class-balance weights so rare diseases (e.g. malaria) are not suppressed

  3. DiseaseRiskModel.predict() takes a user's lat/lon/age/gender and:
       - Finds nearby cases using the grid index
       - Assigns a weight to each case: weight = 1 / distance²
         (closer cases have more influence — this is the KNN / KDE idea)
       - Sums weights per disease → normalizes to get % risk
       - Returns balanced_risk, nearby_cases_25km, density_level

  4. The model can be saved/loaded as a compact JSON file (model_artifact.json).

NOTE: This uses ZERO external libraries — only Python standard library.
      Pure Python so the server starts without any pip installs.
"""

import heapq
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
from xml.etree import ElementTree as ET

# Supported disease names — used throughout the codebase as ground truth labels
DISEASES = ("dengue", "malaria", "typhoid")

_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}

_STANDARD_COLUMNS = {
    "gender": "gender",
    "age": "age",
    "village or ward": "village_or_ward",
    "sub district": "sub_district",
    "district": "district",
    "state": "state",
    "provisional diagnosis": "provisional_diagnosis",
    "confirmed diagnosis": "confirmed_diagnosis",
    "test result": "test_result",
    "pathogen name": "pathogen_name",
    "longitude": "longitude",
    "latitude": "latitude",
}

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _normalize_text(value: str) -> str:
    """Lowercase + strip + collapse whitespace. Used for column name matching."""
    return " ".join((value or "").strip().lower().split())


def _parse_float(value: str) -> Optional[float]:
    """Extract the first number from a string like '32 Years' → 32.0. Returns None if not found."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = _NUM_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _parse_age(value: str) -> Optional[float]:
    """Parse age value and validate it's in the range 0-120. Returns None if invalid."""
    age = _parse_float(value)
    if age is None:
        return None
    if age < 0 or age > 120:
        return None
    return age


def _normalize_gender(value: str) -> str:
    """Normalize gender string to 'male', 'female', 'other', or 'unknown'."""
    text = _normalize_text(value)
    if text in {"male", "m"}:
        return "male"
    if text in {"female", "f"}:
        return "female"
    if text in {"transgender", "trans", "other"}:
        return "other"
    return "unknown"


def _infer_disease(row: Dict[str, str], source_name: str) -> Optional[str]:
    """
    Determine the disease label for a row by scanning the file name,
    provisional_diagnosis, confirmed_diagnosis, and pathogen_name fields.
    Returns 'dengue', 'malaria', 'typhoid', or None if unrecognized.
    """
    blob = " ".join(
        [
            _normalize_text(source_name),
            _normalize_text(row.get("provisional_diagnosis", "")),
            _normalize_text(row.get("confirmed_diagnosis", "")),
            _normalize_text(row.get("pathogen_name", "")),
        ]
    )
    if "dengue" in blob:
        return "dengue"
    if "malaria" in blob:
        return "malaria"
    if "typhoid" in blob or "salmonella" in blob or "paratyphi" in blob:
        return "typhoid"
    return None


def _col_to_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    result = 0
    for ch in letters:
        result = result * 26 + (ord(ch.upper()) - ord("A") + 1)
    return result - 1


def _read_shared_strings(zf: zipfile.ZipFile) -> List[str]:
    path = "xl/sharedStrings.xml"
    if path not in zf.namelist():
        return []

    root = ET.fromstring(zf.read(path))
    out: List[str] = []
    for si in root.findall("main:si", _NS):
        chunks: List[str] = []
        for t_node in si.findall(".//main:t", _NS):
            chunks.append(t_node.text or "")
        out.append("".join(chunks))
    return out


def _cell_text(cell_node: ET.Element, shared_strings: List[str]) -> str:
    c_type = cell_node.attrib.get("t")
    if c_type == "s":
        value_node = cell_node.find("main:v", _NS)
        if value_node is None or value_node.text is None:
            return ""
        try:
            idx = int(value_node.text)
        except ValueError:
            return ""
        if 0 <= idx < len(shared_strings):
            return shared_strings[idx]
        return ""
    if c_type == "inlineStr":
        is_node = cell_node.find("main:is", _NS)
        if is_node is None:
            return ""
        t_node = is_node.find("main:t", _NS)
        return "" if t_node is None or t_node.text is None else t_node.text
    value_node = cell_node.find("main:v", _NS)
    return "" if value_node is None or value_node.text is None else value_node.text


def _workbook_sheets(zf: zipfile.ZipFile) -> List[Tuple[str, str]]:
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))

    rid_to_target: Dict[str, str] = {}
    for rel in rels.findall("rel:Relationship", _NS):
        rel_id = rel.attrib.get("Id")
        target = rel.attrib.get("Target", "")
        if rel_id:
            rid_to_target[rel_id] = target

    out: List[Tuple[str, str]] = []
    for sheet in workbook.findall("main:sheets/main:sheet", _NS):
        name = sheet.attrib.get("name", "sheet")
        rel_id = sheet.attrib.get(
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id",
            "",
        )
        target = rid_to_target.get(rel_id, "")
        if target.startswith("/"):
            path = target.lstrip("/")
        elif target.startswith("xl/"):
            path = target
        else:
            path = f"xl/{target}"
        out.append((name, path))
    return out


def _iter_sheet_rows(
    zf: zipfile.ZipFile, sheet_path: str, shared_strings: List[str]
) -> Iterator[Dict[str, str]]:
    root = ET.fromstring(zf.read(sheet_path))
    sheet_data = root.find("main:sheetData", _NS)
    if sheet_data is None:
        return

    column_map: Dict[int, str] = {}
    header_parsed = False

    for row_node in sheet_data.findall("main:row", _NS):
        parsed_cells: Dict[int, str] = {}
        for c_node in row_node.findall("main:c", _NS):
            ref = c_node.attrib.get("r", "")
            if not ref:
                continue
            idx = _col_to_index(ref)
            parsed_cells[idx] = (_cell_text(c_node, shared_strings) or "").strip()

        if not parsed_cells:
            continue

        if not header_parsed:
            for idx, value in parsed_cells.items():
                key = _STANDARD_COLUMNS.get(_normalize_text(value))
                if key:
                    column_map[idx] = key
            header_parsed = True
            continue

        record: Dict[str, str] = {}
        for idx, value in parsed_cells.items():
            mapped_key = column_map.get(idx)
            if mapped_key:
                record[mapped_key] = value
        if record:
            yield record


def _iter_workbook_rows(workbook_path: Path) -> Iterator[Tuple[str, Dict[str, str]]]:
    with zipfile.ZipFile(workbook_path, "r") as zf:
        shared_strings = _read_shared_strings(zf)
        for sheet_name, sheet_path in _workbook_sheets(zf):
            if sheet_path not in zf.namelist():
                continue
            for row in _iter_sheet_rows(zf, sheet_path, shared_strings):
                row["source_sheet"] = sheet_name
                yield sheet_name, row


def load_cases_from_directory(data_dir: Path) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    files = sorted(data_dir.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No .xlsx files found in {data_dir}")

    cases: List[Dict[str, object]] = []
    source_counter: Counter = Counter()
    disease_counter: Counter = Counter()

    for file_path in files:
        for sheet_name, row in _iter_workbook_rows(file_path):
            lat = _parse_float(row.get("latitude", ""))
            lon = _parse_float(row.get("longitude", ""))
            age = _parse_age(row.get("age", ""))
            if lat is None or lon is None or age is None:
                continue
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                continue

            disease = _infer_disease(row, file_path.name)
            if disease not in DISEASES:
                continue

            case = {
                "latitude": lat,
                "longitude": lon,
                "age": age,
                "gender": _normalize_gender(row.get("gender", "")),
                "disease": disease,
                "district": row.get("district", "").strip(),
                "state": row.get("state", "").strip(),
                "source_file": file_path.name,
                "source_sheet": sheet_name,
            }
            cases.append(case)
            source_counter[f"{file_path.name}:{sheet_name}"] += 1
            disease_counter[disease] += 1

    summary = {
        "total_cases": len(cases),
        "disease_counts": dict(disease_counter),
        "sources": dict(source_counter),
    }
    return cases, summary


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance in kilometres between two GPS coordinates.
    Uses the Haversine formula — accurate for short to medium distances.
    """
    rad = math.pi / 180.0
    p1 = lat1 * rad
    p2 = lat2 * rad
    dlat = (lat2 - lat1) * rad
    dlon = (lon2 - lon1) * rad
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return 6371.0 * c


def _normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Convert raw weighted scores to probabilities that sum to 1.0 (uniform if all zero)."""
    total = sum(scores.values())
    if total <= 0:
        return {disease: round(1.0 / len(DISEASES), 6) for disease in DISEASES}
    return {disease: round(scores.get(disease, 0.0) / total, 6) for disease in DISEASES}


def _percentile(values: List[int], pct: float) -> int:
    if not values:
        return 0
    sorted_vals = sorted(values)
    index = int(round((pct / 100.0) * (len(sorted_vals) - 1)))
    return sorted_vals[index]


class DiseaseRiskModel:
    """
    Weighted K-Nearest Neighbours model for area-level disease risk prediction.

    Training  : Call fit() with a list of case dicts (lat, lon, age, gender, disease).
    Prediction: Call predict() with a user's lat/lon/age/gender to get risk percentages.
    Persistence: save() writes a compact JSON; load() restores from it.

    Parameters:
        cell_size_deg  : Size of spatial grid cells in degrees (~0.25° ≈ 25 km)
        geo_scale_km   : Distance scale for geographic weight calculation
        age_scale_years: Age difference scale for weight calculation
        gender_penalty : Extra distance penalty when gender doesn't match
    """
    def __init__(
        self,
        cell_size_deg: float = 0.25,
        geo_scale_km: float = 45.0,
        age_scale_years: float = 20.0,
        gender_penalty: float = 0.12,
    ):
        self.cell_size_deg = cell_size_deg
        self.geo_scale_km = geo_scale_km
        self.age_scale_years = age_scale_years
        self.gender_penalty = gender_penalty

        self.training_summary: Dict[str, object] = {}
        self._cases: List[Dict[str, object]] = []
        self._disease_counts: Dict[str, int] = {}
        self._class_balance: Dict[str, float] = {}
        self._grid: Dict[Tuple[int, int], List[int]] = {}
        self._cell_counts: Dict[Tuple[int, int], int] = {}
        self._density_thresholds: Dict[str, int] = {}

    def _cell_key(self, latitude: float, longitude: float) -> Tuple[int, int]:
        x = int(math.floor((latitude + 90.0) / self.cell_size_deg))
        y = int(math.floor((longitude + 180.0) / self.cell_size_deg))
        return x, y

    def _build_index(self) -> None:
        grid = defaultdict(list)
        cell_counts = Counter()
        for idx, case in enumerate(self._cases):
            key = self._cell_key(case["latitude"], case["longitude"])
            grid[key].append(idx)
            cell_counts[key] += 1
        self._grid = dict(grid)
        self._cell_counts = dict(cell_counts)

        cell_density_values: List[int] = []
        for key in self._cell_counts:
            cell_density_values.append(self._neighborhood_case_count(key, radius=1))

        self._density_thresholds = {
            "p50": _percentile(cell_density_values, 50),
            "p80": _percentile(cell_density_values, 80),
            "p95": _percentile(cell_density_values, 95),
        }

    def _neighborhood_case_count(self, key: Tuple[int, int], radius: int = 1) -> int:
        total = 0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                total += self._cell_counts.get((key[0] + dx, key[1] + dy), 0)
        return total

    def _candidate_indices(
        self,
        latitude: float,
        longitude: float,
        min_candidates: int = 1600,
        max_cell_radius: int = 6,
    ) -> List[int]:
        if not self._grid:
            return []
        center = self._cell_key(latitude, longitude)
        collected: List[int] = []

        for radius in range(max_cell_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    idxs = self._grid.get((center[0] + dx, center[1] + dy))
                    if idxs:
                        collected.extend(idxs)
            if len(collected) >= min_candidates:
                break

        if collected:
            return collected
        return list(range(len(self._cases)))

    def fit(self, cases: Iterable[Dict[str, object]], training_summary: Optional[Dict[str, object]] = None) -> None:
        self._cases = list(cases)
        if not self._cases:
            raise ValueError("No valid cases were provided.")

        counter = Counter(case["disease"] for case in self._cases)
        self._disease_counts = {d: int(counter.get(d, 0)) for d in DISEASES}

        max_count = max(self._disease_counts.values())
        self._class_balance = {}
        for disease in DISEASES:
            count = self._disease_counts[disease]
            if count <= 0:
                self._class_balance[disease] = 1.0
            else:
                self._class_balance[disease] = math.sqrt(max_count / count)

        self._build_index()
        self.training_summary = training_summary or {}
        self.training_summary.setdefault("total_cases", len(self._cases))
        self.training_summary.setdefault("disease_counts", dict(self._disease_counts))

    def predict(
        self,
        latitude: float,
        longitude: float,
        age: Optional[float] = None,
        gender: Optional[str] = None,
        top_k: int = 1200,
    ) -> Dict[str, object]:
        if not self._cases:
            raise RuntimeError("Model has no training data.")
        if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
            raise ValueError("Latitude/longitude are out of range.")

        normalized_gender = _normalize_gender(gender or "")
        normalized_age = _parse_age(str(age)) if age is not None else None

        candidate_ids = self._candidate_indices(latitude, longitude)
        weighted_samples: List[Tuple[float, str, float, float]] = []

        for idx in candidate_ids:
            case = self._cases[idx]
            geo_km = _haversine_km(
                latitude,
                longitude,
                case["latitude"],
                case["longitude"],
            )
            distance = geo_km / self.geo_scale_km

            if normalized_age is not None:
                distance += abs(case["age"] - normalized_age) / self.age_scale_years

            if normalized_gender != "unknown" and case["gender"] != normalized_gender:
                distance += self.gender_penalty

            distance = max(distance, 1e-6)
            weight = 1.0 / (distance * distance)
            weighted_samples.append((distance, case["disease"], weight, geo_km))

        neighbors = heapq.nsmallest(top_k, weighted_samples, key=lambda x: x[0])

        raw_scores = {disease: 0.0 for disease in DISEASES}
        balanced_scores = {disease: 0.0 for disease in DISEASES}
        nearby_counts = {disease: 0 for disease in DISEASES}

        for _, disease, weight, geo_km in neighbors:
            raw_scores[disease] += weight
            balanced_scores[disease] += weight * self._class_balance.get(disease, 1.0)
            if geo_km <= 25.0:
                nearby_counts[disease] += 1

        raw_distribution = _normalize_scores(raw_scores)
        balanced_distribution = _normalize_scores(balanced_scores)

        cell_key = self._cell_key(latitude, longitude)
        local_density = self._neighborhood_case_count(cell_key, radius=1)
        p50 = self._density_thresholds.get("p50", 0)
        p80 = self._density_thresholds.get("p80", 0)
        p95 = self._density_thresholds.get("p95", 0)

        if local_density >= p95:
            density_level = "very_high"
        elif local_density >= p80:
            density_level = "high"
        elif local_density >= p50:
            density_level = "medium"
        else:
            density_level = "low"

        return {
            "balanced_risk": balanced_distribution,
            "raw_case_distribution": raw_distribution,
            "nearby_cases_25km": nearby_counts,
            "location_density_level": density_level,
            "metadata": {
                "candidates_considered": len(candidate_ids),
                "neighbors_used": len(neighbors),
                "class_balance_factors": {
                    disease: round(self._class_balance.get(disease, 1.0), 4)
                    for disease in DISEASES
                },
            },
            "disclaimer": (
                "Risk is estimated from historical reported positive cases. "
                "This is relative risk, not absolute probability of infection."
            ),
        }

    def to_dict(self) -> Dict[str, object]:
        compact_cases = [
            [
                case["latitude"],
                case["longitude"],
                case["age"],
                case["gender"],
                case["disease"],
            ]
            for case in self._cases
        ]

        return {
            "config": {
                "cell_size_deg": self.cell_size_deg,
                "geo_scale_km": self.geo_scale_km,
                "age_scale_years": self.age_scale_years,
                "gender_penalty": self.gender_penalty,
            },
            "training_summary": self.training_summary,
            "density_thresholds": self._density_thresholds,
            "cases": compact_cases,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "DiseaseRiskModel":
        cfg = payload.get("config", {})
        model = cls(
            cell_size_deg=float(cfg.get("cell_size_deg", 0.25)),
            geo_scale_km=float(cfg.get("geo_scale_km", 45.0)),
            age_scale_years=float(cfg.get("age_scale_years", 20.0)),
            gender_penalty=float(cfg.get("gender_penalty", 0.12)),
        )

        cases = []
        for item in payload.get("cases", []):
            if not isinstance(item, list) or len(item) != 5:
                continue
            cases.append(
                {
                    "latitude": float(item[0]),
                    "longitude": float(item[1]),
                    "age": float(item[2]),
                    "gender": str(item[3]),
                    "disease": str(item[4]),
                }
            )

        model.fit(cases, training_summary=payload.get("training_summary", {}))
        saved_thresholds = payload.get("density_thresholds", {})
        if isinstance(saved_thresholds, dict):
            model._density_thresholds = {
                "p50": int(saved_thresholds.get("p50", model._density_thresholds.get("p50", 0))),
                "p80": int(saved_thresholds.get("p80", model._density_thresholds.get("p80", 0))),
                "p95": int(saved_thresholds.get("p95", model._density_thresholds.get("p95", 0))),
            }
        return model

    def save(self, output_path: Path) -> None:
        payload = self.to_dict()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, separators=(",", ":"))

    @classmethod
    def load(cls, model_path: Path) -> "DiseaseRiskModel":
        with model_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls.from_dict(payload)
