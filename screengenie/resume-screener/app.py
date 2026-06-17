import os
import io
import json
import mimetypes
from datetime import datetime
from urllib.parse import urlparse

import pg8000.dbapi
from flask import Flask, render_template, request, redirect, url_for, g, flash, send_file, abort

from parsing import (
    extract_text_from_file,
    extract_resumes_from_zip,
    top_keywords,
    analyze_resume,
    grade_for_score,
    status_for_score,
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "screen-genie-internal-dev-key")
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024  # 60 MB

# Read at request time so Railway's env vars are always picked up
@app.route("/healthz")
def healthz():
    url = get_database_url()
    has_url = bool(url and "://" in url)
    env_keys = [k for k in os.environ if "DATABASE" in k or "POSTGRES" in k or "PG" in k]
    return {"status": "ok", "has_db_url": has_url, "db_env_keys": env_keys}


def parse_db_url(url):
    # urlparse misreads dotted usernames like postgres.xxxx in Supabase URLs.
    # Manually extract user:password@host:port/db from the URL.
    url = url.strip()
    # strip scheme
    rest = url.split("://", 1)[1]
    # split userinfo from hostinfo
    at_idx = rest.rfind("@")
    userinfo = rest[:at_idx]
    hostinfo = rest[at_idx + 1:]
    # split user:password
    if ":" in userinfo:
        user, password = userinfo.split(":", 1)
    else:
        user, password = userinfo, ""
    # split host:port/db
    if "/" in hostinfo:
        hostport, database = hostinfo.split("/", 1)
    else:
        hostport, database = hostinfo, ""
    if ":" in hostport:
        host, port = hostport.rsplit(":", 1)
        port = int(port)
    else:
        host, port = hostport, 5432
    return {
        "host": host,
        "port": port,
        "database": database.split("?")[0],
        "user": user,
        "password": password,
    }


def get_db():
    if "db" not in g:
        url = get_database_url()
        if not url or "://" not in url:
            raise RuntimeError(f"DATABASE_URL not set or invalid: '{url}'")
        p = parse_db_url(url)
        g.db = pg8000.dbapi.connect(
            host=p["host"], port=p["port"], database=p["database"],
            user=p["user"], password=p["password"], ssl_context=True,
        )
        g.db.autocommit = False
    return g.db


def fetchall_dict(cursor):
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def fetchone_dict(cursor):
    cols = [d[0] for d in cursor.description]
    row = cursor.fetchone()
    return dict(zip(cols, row)) if row else None


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    p = parse_db_url(get_database_url())
    db = pg8000.dbapi.connect(
        host=p["host"], port=p["port"], database=p["database"],
        user=p["user"], password=p["password"], ssl_context=True,
    )
    cur = db.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id SERIAL PRIMARY KEY,
            jd_filename TEXT NOT NULL,
            jd_role TEXT,
            keywords TEXT,
            resume_count INTEGER NOT NULL,
            avg_score REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id SERIAL PRIMARY KEY,
            scan_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            name TEXT,
            email TEXT,
            phone TEXT,
            score INTEGER NOT NULL,
            grade TEXT NOT NULL,
            status TEXT NOT NULL,
            section_scores TEXT,
            matched_keywords TEXT,
            missing_keywords TEXT,
            matched_skills TEXT,
            missing_skills TEXT,
            extra_skills TEXT,
            reasons TEXT,
            file_data BYTEA,
            suggestions TEXT,
            passed_checks TEXT,
            warning_checks TEXT,
            issue_checks TEXT,
            semantic_score INTEGER,
            FOREIGN KEY (scan_id) REFERENCES scans (id)
        )
    """)
    db.commit()
    cur.close()
    db.close()


def guess_role(jd_text, jd_filename):
    for line in jd_text.splitlines():
        line = line.strip()
        if line:
            return line[:80]
    return jd_filename


@app.route("/")
def index():
    return render_template("bulk_screener.html")


@app.route("/scan", methods=["POST"])
def scan():
    jd_file = request.files.get("jd_file")
    resume_mode = request.form.get("resume_mode", "files")

    if not jd_file or jd_file.filename == "":
        flash("Please upload a job description file.")
        return redirect(url_for("index"))

    jd_text = extract_text_from_file(jd_file.filename, jd_file.stream)
    if not jd_text.strip():
        flash("Could not read text from the job description file.")
        return redirect(url_for("index"))

    resume_docs = []

    if resume_mode == "zip":
        zip_file = request.files.get("resume_zip")
        if not zip_file or zip_file.filename == "":
            flash("Please upload a ZIP file of resumes.")
            return redirect(url_for("index"))
        resume_docs = extract_resumes_from_zip(zip_file.stream)
    else:
        files = request.files.getlist("resume_files")
        files = [f for f in files if f and f.filename]
        if not files:
            flash("Please select one or more resume files.")
            return redirect(url_for("index"))
        for f in files:
            raw = f.read()
            text = extract_text_from_file(f.filename, io.BytesIO(raw))
            resume_docs.append((f.filename, text, raw))

    resume_docs = [(name, text, raw) for name, text, raw in resume_docs if text.strip()]
    if not resume_docs:
        flash("No readable resumes were found (check file formats: PDF, DOCX, TXT).")
        return redirect(url_for("index"))

    keywords = top_keywords(jd_text, n=15)
    jd_role = guess_role(jd_text, jd_file.filename)

    candidates = []
    for filename, text, raw in resume_docs:
        analysis = analyze_resume(jd_text, text, keywords)
        score = analysis["score"]
        candidates.append({
            "filename": filename,
            "name": analysis["contact"]["name"],
            "email": analysis["contact"]["email"],
            "phone": analysis["contact"]["phone"],
            "score": score,
            "grade": grade_for_score(score),
            "status": status_for_score(score),
            "section_scores": analysis["section_scores"],
            "matched_keywords": analysis["matched_keywords"],
            "missing_keywords": analysis["missing_keywords"],
            "matched_skills": analysis["matched_skills"],
            "missing_skills": analysis["missing_skills"],
            "extra_skills": analysis["extra_skills"],
            "reasons": analysis["reasons"],
            "suggestions": analysis.get("suggestions", []),
            "passed_checks": analysis.get("passed_checks", []),
            "warning_checks": analysis.get("warning_checks", []),
            "issue_checks": analysis.get("issue_checks", []),
            "semantic_score": analysis.get("semantic_score", 0),
            "raw": raw,
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    avg_score = round(sum(c["score"] for c in candidates) / len(candidates), 1)

    db = get_db()
    cur = db.cursor()
    cur.execute(
        "INSERT INTO scans (jd_filename, jd_role, keywords, resume_count, avg_score, created_at) "
        "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
        (jd_file.filename, jd_role, json.dumps(keywords), len(candidates), avg_score,
         datetime.utcnow().isoformat()),
    )
    scan_id = cur.fetchone()[0]

    for c in candidates:
        cur.execute(
            "INSERT INTO candidates (scan_id, filename, name, email, phone, score, grade, status, "
            "section_scores, matched_keywords, missing_keywords, matched_skills, missing_skills, "
            "extra_skills, reasons, file_data, suggestions, passed_checks, warning_checks, "
            "issue_checks, semantic_score) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (scan_id, c["filename"], c["name"], c["email"], c["phone"], c["score"], c["grade"],
             c["status"], json.dumps(c["section_scores"]), json.dumps(c["matched_keywords"]),
             json.dumps(c["missing_keywords"]), json.dumps(c["matched_skills"]),
             json.dumps(c["missing_skills"]), json.dumps(c["extra_skills"]),
             json.dumps(c["reasons"]), c["raw"],
             json.dumps(c["suggestions"]), json.dumps(c["passed_checks"]),
             json.dumps(c["warning_checks"]), json.dumps(c["issue_checks"]),
             c["semantic_score"]),
        )
    db.commit()
    cur.close()

    return redirect(url_for("results", scan_id=scan_id))


@app.route("/results/<int:scan_id>")
def results(scan_id):
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT id, jd_filename, jd_role, keywords, resume_count, avg_score, created_at FROM scans WHERE id = %s", (scan_id,))
    scan_row = fetchone_dict(cur)
    if not scan_row:
        flash("Scan not found.")
        return redirect(url_for("index"))

    cur.execute(
        "SELECT id, scan_id, filename, name, email, phone, score, grade, status, "
        "section_scores, matched_skills, missing_skills FROM candidates "
        "WHERE scan_id = %s ORDER BY score DESC",
        (scan_id,),
    )
    candidate_rows = fetchall_dict(cur)
    cur.close()

    candidates = []
    for c in candidate_rows:
        c["section_scores"] = json.loads(c["section_scores"])
        c["matched_skills"] = json.loads(c["matched_skills"])
        c["missing_skills"] = json.loads(c["missing_skills"])
        candidates.append(c)

    keywords = json.loads(scan_row["keywords"])
    return render_template("results.html", scan=scan_row, candidates=candidates, keywords=keywords)


@app.route("/candidate/<int:candidate_id>")
def candidate_detail(candidate_id):
    db = get_db()
    cur = db.cursor()
    cur.execute(
        "SELECT id, scan_id, filename, name, email, phone, score, grade, status, "
        "section_scores, matched_keywords, missing_keywords, matched_skills, missing_skills, "
        "extra_skills, reasons, suggestions, passed_checks, warning_checks, issue_checks, "
        "semantic_score, (file_data IS NOT NULL) AS has_file FROM candidates WHERE id = %s",
        (candidate_id,),
    )
    row = fetchone_dict(cur)
    if not row:
        flash("Candidate not found.")
        return redirect(url_for("index"))

    candidate = row
    candidate["section_scores"] = json.loads(candidate["section_scores"])
    candidate["matched_keywords"] = json.loads(candidate["matched_keywords"])
    candidate["missing_keywords"] = json.loads(candidate["missing_keywords"])
    candidate["matched_skills"] = json.loads(candidate["matched_skills"])
    candidate["missing_skills"] = json.loads(candidate["missing_skills"])
    candidate["extra_skills"] = json.loads(candidate["extra_skills"])
    candidate["reasons"] = json.loads(candidate["reasons"])
    candidate["suggestions"] = json.loads(candidate["suggestions"]) if candidate["suggestions"] else []
    candidate["passed_checks"] = json.loads(candidate["passed_checks"]) if candidate["passed_checks"] else []
    candidate["warning_checks"] = json.loads(candidate["warning_checks"]) if candidate["warning_checks"] else []
    candidate["issue_checks"] = json.loads(candidate["issue_checks"]) if candidate["issue_checks"] else []

    cur.execute("SELECT id, jd_filename, jd_role, keywords, resume_count, avg_score, created_at FROM scans WHERE id = %s", (candidate["scan_id"],))
    scan_row = fetchone_dict(cur)
    cur.close()

    return render_template("candidate.html", candidate=candidate, scan=scan_row)


def _send_resume(candidate_id, as_attachment):
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT filename, file_data FROM candidates WHERE id = %s", (candidate_id,))
    row = fetchone_dict(cur)
    cur.close()
    if not row or row["file_data"] is None:
        abort(404)
    mimetype, _ = mimetypes.guess_type(row["filename"])
    file_bytes = bytes(row["file_data"])
    return send_file(
        io.BytesIO(file_bytes),
        mimetype=mimetype or "application/octet-stream",
        as_attachment=as_attachment,
        download_name=row["filename"],
    )


@app.route("/candidate/<int:candidate_id>/view")
def candidate_view_file(candidate_id):
    return _send_resume(candidate_id, as_attachment=False)


@app.route("/candidate/<int:candidate_id>/download")
def candidate_download_file(candidate_id):
    return _send_resume(candidate_id, as_attachment=True)


@app.route("/history")
def history():
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT id, jd_filename, jd_role, keywords, resume_count, avg_score, created_at FROM scans ORDER BY created_at DESC")
    scans = fetchall_dict(cur)
    cur.close()
    return render_template("history.html", scans=scans)


@app.route("/history/<int:scan_id>/delete", methods=["POST"])
def delete_scan(scan_id):
    db = get_db()
    cur = db.cursor()
    cur.execute("DELETE FROM candidates WHERE scan_id = %s", (scan_id,))
    cur.execute("DELETE FROM scans WHERE id = %s", (scan_id,))
    db.commit()
    cur.close()
    flash("Scan deleted.")
    return redirect(url_for("history"))


@app.route("/dashboard")
def dashboard():
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT id, jd_filename, jd_role, keywords, resume_count, avg_score, created_at FROM scans")
    scans = fetchall_dict(cur)
    cur.execute("SELECT id, scan_id, filename, name, score, grade, status FROM candidates")
    candidates = fetchall_dict(cur)

    total_resumes = len(candidates)
    total_scans = len(scans)
    shortlisted = sum(1 for c in candidates if c["status"] == "Shortlist")
    avg_score = round(sum(c["score"] for c in candidates) / total_resumes, 1) if total_resumes else 0

    buckets = {"0-39": 0, "40-59": 0, "60-79": 0, "80-100": 0}
    for c in candidates:
        s = c["score"]
        if s < 40: buckets["0-39"] += 1
        elif s < 60: buckets["40-59"] += 1
        elif s < 80: buckets["60-79"] += 1
        else: buckets["80-100"] += 1

    grade_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for c in candidates:
        grade_counts[c["grade"]] = grade_counts.get(c["grade"], 0) + 1

    cur.execute("SELECT id, scan_id, filename, name, score, grade, status FROM candidates ORDER BY score DESC LIMIT 8")
    top_candidates = fetchall_dict(cur)
    cur.execute("SELECT id, jd_filename, jd_role, keywords, resume_count, avg_score, created_at FROM scans ORDER BY created_at DESC LIMIT 5")
    recent_scans = fetchall_dict(cur)
    cur.close()

    return render_template(
        "dashboard.html",
        total_resumes=total_resumes, total_scans=total_scans,
        shortlisted=shortlisted, avg_score=avg_score,
        buckets=buckets, grade_counts=grade_counts,
        top_candidates=top_candidates, recent_scans=recent_scans,
    )


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
