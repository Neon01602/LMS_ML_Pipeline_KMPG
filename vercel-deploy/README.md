# LMS ML Pipeline — Vercel Deployment

Serves your two trained models (grading + triage) as live API endpoints via
FastAPI on Vercel.

## 1. Export the trained models from your notebook

In your Colab/Jupyter notebook, run cells through **cell 12** (grading) and
**cell 15** (triage) so `winner`, `imputer`, `vec`, `topic_clf`, and
`urgency_clf` exist in memory. Then add `save_models.py` as a new cell and
run it. It produces a `model_artifacts/` folder with 6 files:

```
grading_model.pkl
grading_imputer.pkl
tfidf_vectorizer.pkl
topic_classifier.pkl
urgency_classifier.pkl
metadata.json
```

Download that folder and copy its contents into this project's `models/`
folder (replacing the placeholder `.gitkeep`).

## 2. Project structure

```
vercel-deploy/
├── api/
│   └── index.py        # FastAPI app — /api/triage, /api/grade, /api/health
├── models/              # your 6 exported .pkl/.json files go here
├── requirements.txt
├── vercel.json
└── README.md
```

## 3. Deploy

```bash
npm i -g vercel      # if you don't have the CLI
cd vercel-deploy
vercel               # first deploy, follow the prompts
vercel --prod        # promote to production
```

Or push this folder to a GitHub repo and import it in the Vercel dashboard
(vercel.com/new) — same result, no CLI needed.

## 4. Test live

```bash
curl -X POST https://YOUR-PROJECT.vercel.app/api/triage \
  -H "Content-Type: application/json" \
  -d '{"post_text": "I cannot submit my assignment and the deadline is in 20 minutes, please help urgently"}'

curl -X POST https://YOUR-PROJECT.vercel.app/api/grade \
  -H "Content-Type: application/json" \
  -d '{"code": "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n    return a + b", "pass_rate": 0.9, "test_count": 5}'

curl https://YOUR-PROJECT.vercel.app/api/health
```

## Notes

- **Model size**: LightGBM + scikit-learn objects are usually well under
  Vercel's ~250MB unzipped function limit. The TF-IDF vectorizer is capped
  at `max_features=5000` in the notebook, so it stays small too.
- **`pass_rate` / `test_count`** in the grading pipeline come from an
  automated test harness in the training data, not from the code itself —
  the `/api/grade` endpoint accepts them as optional inputs and falls back
  to median-imputed values (via the saved imputer) if you don't have them
  at request time.
- **Cold starts**: models load once per container and are cached in memory
  (`_state` dict) for subsequent warm requests.
