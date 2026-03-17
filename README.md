# Disease Risk Prediction Starter (Dengue, Malaria, Typhoid)

This project trains a location-aware risk model from your `.xlsx` files and serves predictions for:

- dengue
- malaria
- typhoid

## Important limitation

Your current dataset appears to contain mostly positive confirmed cases.  
Because of that, this model estimates **relative risk from known reported cases**, not absolute population infection probability.

To estimate absolute probability, you will later need:

- negative test records, and/or
- population denominator data by area and time.

## Project files

- `src/disease_risk/engine.py`: data loading, training, and inference engine
- `train_model.py`: builds a model artifact from Excel files
- `predict_cli.py`: quick terminal prediction
- `api_server.py`: backend API + static frontend server + Gemini advice endpoint
- `frontend/index.html`: UI entry point
- `frontend/assets/styles.css`: app styles
- `frontend/assets/app.js`: frontend logic

## Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Run training

```bash
python train_model.py --data-dir . --output model_artifact.json
```

## Configure Gemini API key

Set your key in environment variable `GEMINI_API_KEY`.

Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="YOUR_GEMINI_KEY"
```

Or create a local `.env` file (copy from `.env.example`) in project root.

## Run API server + frontend

```bash
python api_server.py --model model_artifact.json --host 127.0.0.1 --port 8000
```

Open this in browser:

```text
http://127.0.0.1:8000
```

## Optional CLI prediction

```bash
python predict_cli.py --model model_artifact.json --lat 13.0827 --lon 80.2707 --age 28 --gender female
```

## API request examples

`POST /predict` body:

```json
{
  "name": "Aman",
  "age": 28,
  "gender": "male",
  "latitude": 13.0827,
  "longitude": 80.2707
}
```

`POST /advice` body (can include full prediction payload returned by `/predict`):

```json
{
  "input": {
    "name": "Aman",
    "age": 28,
    "gender": "male",
    "latitude": 13.0827,
    "longitude": 80.2707
  },
  "prediction": {
    "balanced_risk": {
      "dengue": 0.44,
      "malaria": 0.39,
      "typhoid": 0.17
    }
  }
}
```

`/predict` response includes:

- `balanced_risk`: disease risk scores adjusted for class imbalance
- `raw_case_distribution`: local case-share without balancing
- `location_density_level`: low / medium / high / very_high
- `disclaimer`: to forward into LLM prompt

`/advice` response includes:

- `advice`: text guidance
- `llm_used`: `true` if Gemini was used, `false` if fallback local advice was used
- `llm_error.hint`: troubleshooting hint when Gemini call fails

## Suggested LLM handoff template

1. Explain the risk in simple language.
2. Give personal precautions for each disease.
3. Mention early warning symptoms.
4. Recommend when to seek medical testing.
5. Include emergency advice and local public-health caution.

## Gemini troubleshooting

If Gemini fails, the app returns fallback advice plus `llm_error.hint`.

Common fixes:

1. Regenerate API key in Google AI Studio and update `.env`.
2. Ensure the key has access to Generative Language API.
3. Try `python api_server.py --model model_artifact.json --gemini-model gemini-2.5-flash`.
4. Restart server after changing `.env`.
