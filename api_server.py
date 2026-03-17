import argparse
import json
import mimetypes
import os
import re
import textwrap
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.disease_risk import DiseaseRiskModel

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
_AVAILABLE_MODELS_CACHE: Dict[str, List[str]] = {}


def _read_json_body(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    content_length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(content_length) if content_length > 0 else b"{}"
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _load_env_file(dotenv_path: Path) -> Dict[str, str]:
    loaded: Dict[str, str] = {}
    if not dotenv_path.exists():
        return loaded
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            loaded[key] = value
    return loaded


def _extract_gemini_text(payload: Dict[str, Any]) -> str:
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        return ""
    first = candidates[0]
    if not isinstance(first, dict):
        return ""
    content = first.get("content", {})
    if not isinstance(content, dict):
        return ""
    parts = content.get("parts", [])
    if not isinstance(parts, list):
        return ""
    chunks = []
    for part in parts:
        if isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())
    return "\n".join(chunks).strip()


def _compact_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=True, separators=(",", ":"))


def _sanitize_advice_text(text: str) -> str:
    cleaned = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"(?m)^\s*---+\s*$", "", cleaned)
    cleaned = cleaned.replace("**", "")
    cleaned = cleaned.replace("__", "")
    cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", cleaned)
    cleaned = cleaned.replace("`", "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    while cleaned.endswith(("*", "_", "-", ":", "`")):
        cleaned = cleaned[:-1].rstrip()
    return cleaned


class GeminiRequestError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        detail: str = "",
        model_name: str = "",
    ):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail
        self.model_name = model_name


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8")
    except Exception:
        return str(exc)


def _is_probable_model_error(status_code: Optional[int], detail: str) -> bool:
    text = detail.lower()
    if status_code in {404}:
        return True
    return any(
        token in text
        for token in (
            "model",
            "not found",
            "not supported",
            "is not available",
            "is not found for api version",
            "unknown model",
            "call listmodels to see the list of available models",
        )
    )


def _gemini_troubleshooting_hint(detail: str) -> str:
    text = detail.lower()
    if "api key not valid" in text or "invalid api key" in text:
        return "API key is invalid. Generate a new key in Google AI Studio and update GEMINI_API_KEY."
    if "permission denied" in text or "access denied" in text:
        return "API key does not have permission for this API/model. Check key restrictions in Google Cloud."
    if "quota" in text or "rate limit" in text:
        return "Quota/rate limit reached. Wait and retry or upgrade API quota."
    if "billing" in text:
        return "Billing may be required or inactive for this API project."
    if "model" in text and "not found" in text:
        return "Requested model is unavailable. Use a listed model such as gemini-2.5-flash."
    if "api has not been used" in text:
        return "Enable Generative Language API in your Google Cloud project."
    return "Check API key validity, model availability, and API restrictions in Google AI Studio/Cloud Console."


def _rule_based_advice(payload: Dict[str, Any]) -> str:
    pred = payload.get("prediction", {})
    if not isinstance(pred, dict):
        pred = {}
    risk = pred.get("balanced_risk", {})
    if not isinstance(risk, dict):
        risk = {}

    ordered = sorted(
        ((str(k), float(v)) for k, v in risk.items() if isinstance(v, (int, float))),
        key=lambda x: x[1],
        reverse=True,
    )
    top = ordered[0][0] if ordered else "disease"

    if top == "dengue":
        focus = "avoid mosquito bites (nets, repellents, full sleeves, no stagnant water)"
    elif top == "malaria":
        focus = "strict mosquito protection, especially at dusk/night"
    elif top == "typhoid":
        focus = "safe drinking water, hand hygiene, and clean food handling"
    else:
        focus = "general hygiene and early testing if symptoms appear"

    return (
        "Gemini response unavailable, so this is a local fallback.\n\n"
        f"Highest relative local risk appears to be: {top}.\n"
        f"Main prevention focus: {focus}.\n"
        "Watch for fever, fatigue, persistent headache, vomiting, or dehydration.\n"
        "Seek medical testing quickly if symptoms continue for more than 24-48 hours."
    )


def _build_gemini_prompt(payload: Dict[str, Any]) -> str:
    user_input = payload.get("input", {})
    if not isinstance(user_input, dict):
        user_input = {}
    prediction = payload.get("prediction", {})
    if not isinstance(prediction, dict):
        prediction = {}

    return textwrap.dedent(
        f"""
        You are a health risk guidance assistant.
        Give practical, clear, non-alarming advice based on relative local disease risk.
        Do not claim certainty or diagnosis.
        Mention that final diagnosis needs a licensed doctor and test.
        Use plain text only. Do not use markdown symbols such as **, #, ---, or bullets with *.
        Keep the answer concise (about 140-220 words).

        Return output in this structure:
        1. Quick Risk Summary
        2. Most Important Precautions Today
        3. Early Symptoms To Monitor (dengue/malaria/typhoid)
        4. When To Test Or Visit Hospital
        5. Safety Notes

        User profile JSON:
        {_compact_json(user_input)}

        Model output JSON:
        {_compact_json(prediction)}
        """
    ).strip()


def _call_gemini_once(api_key: str, model_name: str, prompt: str) -> str:
    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent?key={api_key}"
    )
    generation_config: Dict[str, Any] = {
        "temperature": 0.3,
        "maxOutputTokens": 1800,
        "responseMimeType": "text/plain",
    }

    lowered = model_name.lower()
    if "gemini-2.5" in lowered or lowered.startswith("gemini-3"):
        generation_config["thinkingConfig"] = {"thinkingBudget": 0}

    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=40) as response:
            raw = response.read().decode("utf-8")
        parsed = json.loads(raw)
        text = _extract_gemini_text(parsed)
        if not text:
            raise GeminiRequestError(
                "Gemini returned an empty response.",
                detail=raw[:700],
                model_name=model_name,
            )
        return _sanitize_advice_text(text)
    except urllib.error.HTTPError as exc:
        detail = _http_error_detail(exc)
        raise GeminiRequestError(
            "Gemini API request failed",
            status_code=exc.code,
            detail=detail[:1200],
            model_name=model_name,
        ) from exc
    except urllib.error.URLError as exc:
        raise GeminiRequestError(
            "Network error while calling Gemini",
            detail=str(exc),
            model_name=model_name,
        ) from exc


def _normalize_model_name(name: str) -> str:
    clean = (name or "").strip()
    if clean.startswith("models/"):
        clean = clean.split("/", 1)[1]
    return clean


def _is_text_generation_model(name: str) -> bool:
    text = name.lower()
    if not text.startswith("gemini"):
        return False
    blocked_tokens = (
        "image",
        "tts",
        "robotics",
        "computer-use",
        "customtools",
    )
    return not any(token in text for token in blocked_tokens)


def _model_priority(name: str) -> int:
    text = name.lower()
    if "gemini-2.5-flash" in text and "lite" not in text and "preview" not in text:
        return 0
    if "gemini-2.0-flash" in text and "lite" not in text and "preview" not in text:
        return 1
    if "gemini-flash-latest" in text:
        return 2
    if "flash" in text and "lite" not in text and "preview" not in text:
        return 3
    if "gemini-2.5-pro" in text and "preview" not in text:
        return 4
    if "gemini-pro-latest" in text:
        return 5
    if "pro" in text and "preview" not in text:
        return 6
    if "flash-lite" in text:
        return 7
    return 20


def _fetch_available_models(api_key: str) -> List[str]:
    cached = _AVAILABLE_MODELS_CACHE.get(api_key)
    if cached is not None:
        return cached

    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=25) as response:
        raw = response.read().decode("utf-8")
    payload = json.loads(raw)

    models: List[str] = []
    for item in payload.get("models", []):
        if not isinstance(item, dict):
            continue
        methods = item.get("supportedGenerationMethods", [])
        if not isinstance(methods, list) or "generateContent" not in methods:
            continue
        name = _normalize_model_name(str(item.get("name", "")))
        if name and _is_text_generation_model(name):
            models.append(name)

    unique_models = sorted(set(models), key=lambda m: (_model_priority(m), m))
    _AVAILABLE_MODELS_CACHE[api_key] = unique_models
    return unique_models


def _build_model_candidates(api_key: str, preferred_model: str) -> List[str]:
    candidates: List[str] = []

    def add(name: str) -> None:
        normalized = _normalize_model_name(name)
        if not normalized:
            return
        if normalized in candidates:
            return
        if _is_text_generation_model(normalized):
            candidates.append(normalized)

    seed_models = (
        preferred_model,
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-flash-latest",
        "gemini-2.5-pro",
        "gemini-pro-latest",
    )
    for model in seed_models:
        add(model)

    try:
        for model in _fetch_available_models(api_key):
            add(model)
    except Exception:
        pass

    return candidates[:12]


def _should_try_next_model(exc: GeminiRequestError) -> bool:
    if _is_probable_model_error(exc.status_code, exc.detail):
        return True
    if exc.status_code in {429, 500, 502, 503, 504}:
        return True
    detail = exc.detail.lower()
    transient_tokens = ("rate limit", "quota", "temporar", "backend error", "unavailable")
    return any(token in detail for token in transient_tokens)


def _call_gemini_with_fallback(
    api_key: str,
    preferred_model: str,
    prompt: str,
) -> Tuple[str, str]:
    candidates = _build_model_candidates(api_key, preferred_model)
    last_error: Optional[GeminiRequestError] = None
    for model_name in candidates:
        try:
            return _call_gemini_once(api_key, model_name, prompt), model_name
        except GeminiRequestError as exc:
            last_error = exc
            if not _should_try_next_model(exc):
                break

    if last_error is None:
        raise GeminiRequestError("Gemini API request failed with unknown error.")
    raise last_error


class PredictHandler(BaseHTTPRequestHandler):
    model: DiseaseRiskModel = None  # type: ignore[assignment]
    gemini_api_key: str = ""
    gemini_model_name: str = "gemini-2.5-flash"

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, file_path: Path) -> None:
        if not file_path.exists() or not file_path.is_file():
            self._send_json(404, {"error": "Not found"})
            return
        data = file_path.read_bytes()
        content_type, _ = mimetypes.guess_type(str(file_path))
        self.send_response(200)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _frontend_path(self, raw_path: str) -> Optional[Path]:
        clean = raw_path.split("?", 1)[0]
        if clean == "/":
            return FRONTEND_DIR / "index.html"
        if clean.startswith("/assets/"):
            rel = clean.lstrip("/")
            target = (FRONTEND_DIR / rel).resolve()
            if FRONTEND_DIR.resolve() in target.parents:
                return target
        return None

    def _predict_from_payload(self, body: Dict[str, Any]) -> Dict[str, Any]:
        latitude = body.get("latitude", body.get("lat"))
        longitude = body.get("longitude", body.get("lon"))
        age = body.get("age")
        gender = body.get("gender", "unknown")
        name = body.get("name", "")

        if latitude is None or longitude is None:
            raise ValueError("latitude and longitude are required")

        result = self.model.predict(
            latitude=float(latitude),
            longitude=float(longitude),
            age=None if age is None else float(age),
            gender=str(gender),
        )

        return {
            "input": {
                "name": name,
                "age": age,
                "gender": gender,
                "latitude": latitude,
                "longitude": longitude,
            },
            "prediction": result,
        }

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send_json(200, {"ok": True})

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(
                200,
                {
                    "ok": True,
                    "model_cases": self.model.training_summary.get("total_cases", 0),
                },
            )
            return
        static_target = self._frontend_path(self.path)
        if static_target is not None:
            self._send_file(static_target)
            return
        self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/predict", "/advice"}:
            self._send_json(404, {"error": "Not found"})
            return

        try:
            body = _read_json_body(self)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON body"})
            return

        try:
            if self.path == "/predict":
                response_payload = self._predict_from_payload(body)
                self._send_json(200, response_payload)
                return

            composite = body
            if "prediction" not in composite:
                composite = self._predict_from_payload(body)

            if not self.gemini_api_key:
                self._send_json(
                    200,
                    {
                        "advice": _rule_based_advice(composite),
                        "llm_used": False,
                        "reason": "GEMINI_API_KEY is not configured",
                    },
                )
                return

            prompt = _build_gemini_prompt(composite)
            advice_text, used_model = _call_gemini_with_fallback(
                api_key=self.gemini_api_key,
                preferred_model=self.gemini_model_name,
                prompt=prompt,
            )

            self._send_json(
                200,
                {
                    "advice": advice_text,
                    "llm_used": True,
                    "model": used_model,
                },
            )
            return
        except GeminiRequestError as exc:
            self._send_json(
                200,
                {
                    "advice": _rule_based_advice(composite),
                    "llm_used": False,
                    "reason": "Gemini API request failed; local fallback used",
                    "llm_error": {
                        "message": str(exc),
                        "status_code": exc.status_code,
                        "model": exc.model_name,
                        "detail": exc.detail[:600],
                        "hint": _gemini_troubleshooting_hint(exc.detail),
                    },
                },
            )
            return
        except Exception as exc:
            self._send_json(400, {"error": str(exc)})
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve disease risk prediction API.")
    parser.add_argument("--model", type=Path, default=Path("model_artifact.json"))
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model name for /advice endpoint.",
    )
    args = parser.parse_args()

    env_file_vars = _load_env_file(BASE_DIR / ".env")
    for env_key, env_value in env_file_vars.items():
        os.environ.setdefault(env_key, env_value)

    model = DiseaseRiskModel.load(args.model)
    PredictHandler.model = model
    PredictHandler.gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
    PredictHandler.gemini_model_name = args.gemini_model.strip() or "gemini-2.5-flash"

    server = ThreadingHTTPServer((args.host, args.port), PredictHandler)
    print(f"Serving on http://{args.host}:{args.port}")
    print("Endpoints:")
    print("  GET  /health")
    print("  GET  /")
    print("  POST /predict")
    print("  POST /advice")
    if PredictHandler.gemini_api_key:
        print(f"Gemini enabled with model: {PredictHandler.gemini_model_name}")
    else:
        print("Gemini disabled (set GEMINI_API_KEY to enable /advice)")
    server.serve_forever()


if __name__ == "__main__":
    main()
