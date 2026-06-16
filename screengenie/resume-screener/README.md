# Screen Genie — P&C Bulk Resume Screener

A working internal web app that screens bulk resumes against a job description
using TF-IDF semantic similarity + JD-derived keyword coverage. Domain-agnostic
(works for tech, sales, finance, ops, etc.) because keywords are auto-extracted
from the JD itself rather than being hardcoded.

## Features

1. **Bulk Screener** — Upload one JD (PDF / DOCX / TXT) and many resumes (multiple
   files or a single ZIP). Parses each file, extracts contact info (name, email,
   phone) and scores every resume.
2. **Scan History** — Every screening run is persisted in SQLite; open any past
   scan to see its ranked board, or delete it.
3. **Recruiter Dashboard** — Aggregate KPIs (total resumes, shortlisted, avg score,
   bulk scans), score distribution, grade mix, top candidates, recent scans.

Per-candidate detail page shows the score breakdown (semantic vs keyword),
matched / missing keywords, contact details, grade and status.

## Run locally

```bash
pip install -r requirements.txt
python app.py
# open http://localhost:5000
```

The SQLite database `screengenie.db` is created automatically on first run.

## Deploy

The app is a standard Flask WSGI application (`app:app`).

### Option 1: Gunicorn + any Linux host (Render / Railway / VPS)

```bash
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:8000 app:app
```

Make sure the runtime directory is writable so `screengenie.db` can be created
(or set a persistent disk on Render / Railway / Fly.io).

### Option 2: Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn
COPY . .
EXPOSE 8000
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "app:app"]
```

```bash
docker build -t screen-genie .
docker run -p 8000:8000 -v $(pwd)/data:/app/data screen-genie
```

### Option 3: PythonAnywhere / Heroku-style

`Procfile`:
```
web: gunicorn app:app
```

## How scoring works

For each (JD, resume) pair:

- **Semantic match**: TF-IDF vectorizer (English stopwords) over the two
  documents, then cosine similarity → 0-100%.
- **Keyword coverage**: The top 15 most frequent non-stopword terms in the JD
  are taken as required keywords. Coverage = matched / 15.
- **Overall score = 0.55 × semantic + 0.45 × coverage**, clamped to 0-100.
- **Grade**: A ≥ 80, B ≥ 65, C ≥ 50, D < 50.
- **Status**: Shortlist ≥ 70, Review ≥ 45, Reject < 45.

Tweak the weights and thresholds in `parsing.py` (`score_resume`,
`grade_for_score`, `status_for_score`) to match your team's rubric.

## File layout

```
app.py             Flask routes + DB
parsing.py         PDF/DOCX/ZIP parsing, contact extraction, scoring
templates/         Jinja2 templates (base, bulk_screener, results, candidate, history, dashboard)
static/style.css   UI (dark glass theme)
requirements.txt
```
