"""
app.py — Web UI for GNR Building Extraction Pipeline
=====================================================
Now accepts any image uploaded from the user's device.
Run with:  python app.py
Then open: http://localhost:5000
"""

import os
import sys
import json
import base64
import threading
import subprocess
import uuid
import time
import shutil
import zipfile
import glob
from flask import Flask, jsonify, send_file, request, render_template_string
from pathlib import Path
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 256 * 1024 * 1024   # 256 MB upload limit (5 Sentinel TIFFs)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
STAGES_DIR  = os.path.join(OUTPUTS_DIR, "stages")
SAT_DIR     = os.path.join(BASE_DIR, "sat_image")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")

for d in (OUTPUTS_DIR, STAGES_DIR, UPLOADS_DIR):
    os.makedirs(d, exist_ok=True)

ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}

# Sentinel-2 bands required to run the pipeline.
# B08 (NIR) + B11 (SWIR) drive the true NDBI; for the RGB composite we need
# either a True_color TIFF or B02/B03/B04 individually.
SENTINEL_REQUIRED_NDBI_BANDS = ('B08', 'B11')
SENTINEL_RGB_BANDS           = ('B02', 'B03', 'B04')


def _glob_band(d, band):
    """Find the first TIFF in d (recursive) that looks like a given Sentinel band."""
    for pattern in (f'**/*_{band}_*.tif*', f'**/*{band}_*.tif*', f'**/*{band}*.tif*'):
        matches = glob.glob(os.path.join(d, pattern), recursive=True)
        if matches:
            return matches[0]
    return None


def validate_sentinel_dir(d):
    """
    Check that d holds the bands the pipeline needs.
    Returns (ok: bool, missing: list[str]).
    """
    missing = []
    for band in SENTINEL_REQUIRED_NDBI_BANDS:
        if _glob_band(d, band) is None:
            missing.append(band)

    has_true_color = bool(glob.glob(os.path.join(d, '**/*True_color*.tif*'), recursive=True))
    if not has_true_color:
        for band in SENTINEL_RGB_BANDS:
            if _glob_band(d, band) is None:
                missing.append(band)

    return (len(missing) == 0, missing)

# ── Job registry ───────────────────────────────────────────────────────────────
jobs      = {}
jobs_lock = threading.Lock()


def allowed(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def img_to_b64(path):
    if not path or not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        data = f.read()
    ext  = Path(path).suffix.lstrip(".").lower()
    mime = "png" if ext == "png" else ("tiff" if ext in ("tif","tiff") else "jpeg")
    return "data:image/{};base64,{}".format(mime, base64.b64encode(data).decode())


def run_job(job_id, script_args, input_image_path):
    with jobs_lock:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["log"]    = ""

    try:
        proc = subprocess.Popen(
            script_args, cwd=BASE_DIR,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        log_lines = []
        for line in proc.stdout:
            log_lines.append(line)
            with jobs_lock:
                jobs[job_id]["log"] = "".join(log_lines)
        proc.wait()

        if proc.returncode != 0:
            raise RuntimeError("Process exited with code {}".format(proc.returncode))

        stage_names = [
            (1, "original",          "1. Original"),
            (2, "ndbi",              "2. NDBI Map"),
            (3, "builtup_mask",      "3. Built-up Mask"),
            (4, "suppressed_gray",   "4. Suppressed Gray"),
            (5, "canny_edges",       "5. Canny Edges"),
            (6, "after_morphology",  "6. After Morphology"),
            (7, "hough_accumulator", "7. Hough Accumulator"),
            (8, "detected_outlines", "8. Detected Outlines"),
        ]
        stages = []
        for idx, key, label in stage_names:
            p   = os.path.join(STAGES_DIR, "{}_{}.png".format(idx, key))
            b64 = img_to_b64(p)
            if b64:
                stages.append({"label": label, "data": b64})

        with jobs_lock:
            jobs[job_id]["status"]       = "done"
            jobs[job_id]["input_img"]    = img_to_b64(input_image_path)
            jobs[job_id]["output_img"]   = img_to_b64(
                os.path.join(OUTPUTS_DIR, "final_outlines.png"))
            jobs[job_id]["pipeline_img"] = img_to_b64(
                os.path.join(OUTPUTS_DIR, "building_extraction_pipeline.png"))
            jobs[job_id]["stages"]       = stages

    except Exception as e:
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"]  = str(e)

    finally:
        tmp = jobs.get(job_id, {}).get("_tmp_path")
        if tmp:
            try:
                if os.path.isfile(tmp):
                    os.remove(tmp)
                elif os.path.isdir(tmp):
                    shutil.rmtree(tmp, ignore_errors=True)
            except OSError:
                pass


# ── Hyperparameter helpers ──────────────────────────────────────────────────────

DEFAULTS = {
    "gauss_ksize":    "5",
    "gauss_sigma":    "1.4",
    "morph_shape":    "square",
    "morph_size":     "3",
    "ndbi_threshold": "0.1",
}

def build_script_args(python, script, image_path, params):
    """Build CLI args list for building_extraction.py with hyperparams."""
    def _get(key):
        val = params.get(key, "").strip() if params else ""
        return val if val else DEFAULTS[key]

    gauss_ksize    = _get("gauss_ksize")
    gauss_sigma    = _get("gauss_sigma")
    morph_shape    = _get("morph_shape")
    morph_size     = _get("morph_size")
    ndbi_threshold = _get("ndbi_threshold")

    return [
        python, script, image_path,
        "--gauss-ksize",    gauss_ksize,
        "--gauss-sigma",    gauss_sigma,
        "--morph-shape",    morph_shape,
        "--morph-size",     morph_size,
        "--ndbi-threshold", ndbi_threshold,
    ]


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/run_upload", methods=["POST"])
def run_upload():
    if "image" not in request.files:
        return jsonify({"error": "No file part named 'image'"}), 400
    f = request.files["image"]
    if f.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed(f.filename):
        return jsonify({"error": "Unsupported file type. Allowed: " +
                        ", ".join(ALLOWED_EXTENSIONS)}), 400

    safe_name = "{}_{}".format(uuid.uuid4().hex, secure_filename(f.filename))
    tmp_path  = os.path.join(UPLOADS_DIR, safe_name)
    f.save(tmp_path)

    job_id      = str(uuid.uuid4())
    script_args = build_script_args(sys.executable, "building_extraction.py",
                                    tmp_path, request.form)
    label       = Path(f.filename).name

    with jobs_lock:
        jobs[job_id] = {
            "status": "queued", "label": label,
            "log": "", "input_img": None, "output_img": None,
            "pipeline_img": None, "stages": [], "error": None,
            "_tmp_path": tmp_path,
        }

    threading.Thread(target=run_job,
                     args=(job_id, script_args, tmp_path),
                     daemon=True).start()
    return jsonify({"job_id": job_id, "label": label})


@app.route("/run_sentinel", methods=["POST"])
def run_sentinel():
    if not os.path.isdir(SAT_DIR):
        return jsonify({"error":
            "sat_image/ folder not found. Either place Sentinel-2 TIFFs there "
            "or upload your own bands using the upload zone."}), 404

    ok, missing = validate_sentinel_dir(SAT_DIR)
    if not ok:
        return jsonify({"error":
            "sat_image/ is missing required band(s): " + ", ".join(missing) +
            ". Need B08 + B11 plus either a True_color TIFF or B02 + B03 + B04."
        }), 400

    params      = request.get_json(silent=True) or {}
    job_id      = str(uuid.uuid4())
    script_args = build_script_args(sys.executable, "sentinel_pipeline.py",
                                    SAT_DIR, params)
    input_path  = os.path.join(OUTPUTS_DIR, "sentinel2_rgb.png")
    label       = "Sentinel-2 (sample data)"

    with jobs_lock:
        jobs[job_id] = {
            "status": "queued", "label": label,
            "log": "", "input_img": None, "output_img": None,
            "pipeline_img": None, "stages": [], "error": None,
            "_tmp_path": None,
        }

    threading.Thread(target=run_job,
                     args=(job_id, script_args, input_path),
                     daemon=True).start()
    return jsonify({"job_id": job_id, "label": label})


@app.route("/run_sentinel_upload", methods=["POST"])
def run_sentinel_upload():
    """
    Accept either:
      - multiple .tif/.tiff files (typical: B02, B03, B04, B08, B11), OR
      - a single .zip containing those TIFFs (auto-extracted, nested folders OK).
    Files are written to a per-job temp dir which is cleaned up after the run.
    """
    files = request.files.getlist("bands")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No files uploaded"}), 400

    job_id  = str(uuid.uuid4())
    tmp_dir = os.path.join(UPLOADS_DIR, "sentinel_" + job_id)
    os.makedirs(tmp_dir, exist_ok=True)

    saved_count = 0
    try:
        for f in files:
            if f.filename == "":
                continue
            name = secure_filename(f.filename)
            ext  = Path(name).suffix.lower()

            if ext == ".zip":
                zpath = os.path.join(tmp_dir, name)
                f.save(zpath)
                try:
                    with zipfile.ZipFile(zpath) as zf:
                        # Reject zips containing absolute paths or '..' to be safe
                        for member in zf.namelist():
                            if os.path.isabs(member) or ".." in Path(member).parts:
                                shutil.rmtree(tmp_dir, ignore_errors=True)
                                return jsonify({"error":
                                    f"Refusing to extract unsafe path in zip: {member}"
                                }), 400
                        zf.extractall(tmp_dir)
                except zipfile.BadZipFile:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    return jsonify({"error": f"Could not unzip {name}"}), 400
                os.remove(zpath)
                # Count extracted TIFFs
                saved_count += sum(1 for _ in glob.glob(
                    os.path.join(tmp_dir, '**/*.tif*'), recursive=True))

            elif ext in (".tif", ".tiff"):
                f.save(os.path.join(tmp_dir, name))
                saved_count += 1

            else:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return jsonify({"error":
                    f"Unsupported file: {name}. Need .tif/.tiff or a single .zip."
                }), 400
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return jsonify({"error": "Upload failed: " + str(e)}), 500

    if saved_count == 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return jsonify({"error": "No TIFFs found in the upload."}), 400

    ok, missing = validate_sentinel_dir(tmp_dir)
    if not ok:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return jsonify({"error":
            "Missing required Sentinel-2 band(s): " + ", ".join(missing) +
            ". Need B08 + B11 plus either a True_color TIFF or B02 + B03 + B04."
        }), 400

    params      = request.form
    script_args = build_script_args(sys.executable, "sentinel_pipeline.py",
                                    tmp_dir, params)
    input_path  = os.path.join(OUTPUTS_DIR, "sentinel2_rgb.png")
    label       = f"Sentinel-2 (uploaded · {saved_count} files)"

    with jobs_lock:
        jobs[job_id] = {
            "status": "queued", "label": label,
            "log": "", "input_img": None, "output_img": None,
            "pipeline_img": None, "stages": [], "error": None,
            "_tmp_path": tmp_dir,
        }

    threading.Thread(target=run_job,
                     args=(job_id, script_args, input_path),
                     daemon=True).start()
    return jsonify({"job_id": job_id, "label": label})


@app.route("/poll/<job_id>")
def poll(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify({k: v for k, v in job.items() if not k.startswith("_")})



# ── HTML Template ───────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>GNR Building Extraction</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;600;900&family=Exo+2:wght@300;400;600&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg:#050a0f; --surface:#0a1520; --surface2:#0f1e2e; --border:#1a3a5c;
    --accent:#00d4ff; --accent2:#00ff9d; --accent3:#ff6b35;
    --text:#c8dff0; --text-dim:#4a7a9b;
    --glow:0 0 20px rgba(0,212,255,0.3);
    --glow-green:0 0 20px rgba(0,255,157,0.3);
    --glow-orange:0 0 20px rgba(255,107,53,0.3);
    --mono:'Share Tech Mono',monospace;
    --display:'Orbitron',sans-serif;
    --body:'Exo 2',sans-serif;
  }
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:var(--body);
    min-height:100vh;overflow-x:hidden;
    background-image:linear-gradient(rgba(0,212,255,.025) 1px,transparent 1px),
      linear-gradient(90deg,rgba(0,212,255,.025) 1px,transparent 1px);
    background-size:40px 40px}
  body::after{content:'';position:fixed;inset:0;pointer-events:none;z-index:9999;
    background:repeating-linear-gradient(0deg,transparent,transparent 2px,
      rgba(0,0,0,.07) 2px,rgba(0,0,0,.07) 4px)}

  /* Header */
  header{border-bottom:1px solid var(--border);padding:22px 40px;
    display:flex;align-items:center;gap:20px;
    background:linear-gradient(180deg,rgba(0,212,255,.05) 0%,transparent 100%);
    position:relative;overflow:hidden}
  header::before{content:'';position:absolute;inset:0;
    background:linear-gradient(90deg,transparent,rgba(0,212,255,.04),transparent);
    animation:scan 5s linear infinite}
  @keyframes scan{from{transform:translateX(-100%)}to{transform:translateX(100%)}}
  .logo-icon{width:48px;height:48px;border:2px solid var(--accent);border-radius:8px;
    display:flex;align-items:center;justify-content:center;
    font-family:var(--display);font-size:16px;font-weight:900;color:var(--accent);
    box-shadow:var(--glow),inset 0 0 12px rgba(0,212,255,.1);position:relative;z-index:1;flex-shrink:0}
  h1{font-family:var(--display);font-size:20px;font-weight:900;color:var(--accent);
    text-transform:uppercase;letter-spacing:3px;text-shadow:0 0 12px rgba(0,212,255,.5)}
  .header-sub{font-family:var(--mono);font-size:10.5px;color:var(--text-dim);letter-spacing:2px;margin-top:4px}
  .header-text{position:relative;z-index:1}
  .badge{margin-left:auto;font-family:var(--mono);font-size:10px;color:var(--accent2);
    border:1px solid rgba(0,255,157,.3);padding:5px 12px;border-radius:3px;
    position:relative;z-index:1;text-transform:uppercase;letter-spacing:1px}

  /* Layout */
  main{max-width:1400px;margin:0 auto;padding:36px 40px}

  .panel{background:var(--surface);border:1px solid var(--border);border-radius:12px;
    padding:32px;margin-bottom:28px;position:relative;overflow:hidden}
  .corner{position:absolute;width:14px;height:14px;border-color:var(--accent);border-style:solid;opacity:.4}
  .tl{top:8px;left:8px;border-width:2px 0 0 2px}
  .tr{top:8px;right:8px;border-width:2px 2px 0 0}
  .bl{bottom:8px;left:8px;border-width:0 0 2px 2px}
  .br{bottom:8px;right:8px;border-width:0 2px 2px 0}

  .panel-title{font-family:var(--display);font-size:11px;font-weight:600;
    color:var(--accent);text-transform:uppercase;letter-spacing:3px;
    margin-bottom:24px;display:flex;align-items:center;gap:10px}
  .panel-title::before{content:'';width:16px;height:2px;background:var(--accent);box-shadow:var(--glow)}

  /* Mode tabs */
  .mode-tabs{display:flex;border:1px solid var(--border);border-radius:8px;
    overflow:hidden;margin-bottom:28px}
  .mode-tab{flex:1;padding:13px 20px;cursor:pointer;font-family:var(--mono);
    font-size:12px;letter-spacing:1px;text-transform:uppercase;text-align:center;
    background:transparent;border:none;color:var(--text-dim);transition:all .2s}
  .mode-tab:not(:last-child){border-right:1px solid var(--border)}
  .mode-tab.active{background:rgba(0,212,255,.1);color:var(--accent);
    box-shadow:inset 0 -2px 0 var(--accent)}
  .mode-tab:hover:not(.active){color:var(--text);background:rgba(255,255,255,.03)}

  /* Upload zone */
  .upload-zone{border:2px dashed var(--border);border-radius:10px;padding:52px 32px;
    text-align:center;cursor:pointer;transition:all .25s;position:relative;
    background:rgba(0,0,0,.2)}
  .upload-zone:hover,.upload-zone.drag-over{border-color:var(--accent);
    background:rgba(0,212,255,.05);box-shadow:var(--glow)}
  .upload-zone input[type=file]{position:absolute;inset:0;opacity:0;
    cursor:pointer;width:100%;height:100%}
  .up-icon{font-size:46px;margin-bottom:14px;opacity:.55}
  .up-title{font-family:var(--display);font-size:13px;font-weight:600;
    color:var(--accent);letter-spacing:2px;text-transform:uppercase;margin-bottom:8px}
  .up-hint{font-family:var(--mono);font-size:11px;color:var(--text-dim);letter-spacing:1px}
  .up-hint em{color:var(--accent2);font-style:normal}

  /* File preview */
  .file-preview{display:none;margin-top:20px;background:var(--bg);
    border:1px solid var(--border);border-radius:8px;padding:14px 18px;
    align-items:center;gap:16px}
  .file-preview.visible{display:flex}
  .prev-thumb{width:62px;height:62px;object-fit:cover;border-radius:5px;
    border:1px solid var(--border)}
  .prev-info{flex:1}
  .prev-name{font-family:var(--mono);font-size:13px;color:var(--text);word-break:break-all}
  .prev-meta{font-family:var(--mono);font-size:11px;color:var(--text-dim);margin-top:4px}
  .prev-clear{background:none;border:1px solid var(--border);border-radius:5px;
    color:var(--text-dim);font-size:12px;padding:5px 10px;cursor:pointer;
    font-family:var(--mono);transition:all .15s}
  .prev-clear:hover{color:#ff4444;border-color:rgba(255,68,68,.5)}

  /* Sentinel info */
  .sentinel-info{display:flex;gap:20px;align-items:flex-start;background:var(--bg);
    border:1px solid var(--border);border-radius:10px;padding:22px 24px}
  .sat-icon{font-size:38px;flex-shrink:0;margin-top:2px;opacity:.8}
  .sat-det h3{font-family:var(--display);font-size:12px;font-weight:600;
    color:var(--accent2);text-transform:uppercase;letter-spacing:2px;margin-bottom:8px}
  .sat-det p{font-family:var(--mono);font-size:11.5px;color:var(--text-dim);line-height:1.75}
  .sat-det p em{color:var(--accent);font-style:normal}
  .band-pills{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
  .band-pill{font-family:var(--mono);font-size:10px;padding:3px 9px;border-radius:3px;
    background:rgba(0,212,255,.08);border:1px solid rgba(0,212,255,.25);
    color:var(--accent);letter-spacing:1px}

  /* Run row */
  .run-row{display:flex;gap:14px;align-items:center;margin-top:24px;flex-wrap:wrap}
  .run-btn{background:linear-gradient(135deg,rgba(0,212,255,.15),rgba(0,212,255,.05));
    border:1px solid var(--accent);border-radius:6px;color:var(--accent);
    font-family:var(--display);font-size:12px;font-weight:600;letter-spacing:2px;
    text-transform:uppercase;padding:13px 36px;cursor:pointer;transition:all .2s;
    position:relative;overflow:hidden}
  .run-btn:hover:not(:disabled){background:linear-gradient(135deg,rgba(0,212,255,.3),rgba(0,212,255,.1));
    box-shadow:var(--glow);transform:translateY(-1px)}
  .run-btn:disabled{opacity:.35;cursor:not-allowed;transform:none}

  /* Status */
  .status-bar{flex:1;min-width:220px;background:var(--bg);border:1px solid var(--border);
    border-radius:6px;padding:12px 16px;font-family:var(--mono);font-size:12px;
    display:flex;align-items:center;gap:10px}
  .sdot{width:8px;height:8px;border-radius:50%;background:var(--text-dim);flex-shrink:0}
  .sdot.running{background:var(--accent3);animation:blink .8s ease-in-out infinite}
  .sdot.done{background:var(--accent2)}
  .sdot.error{background:#ff4444}
  @keyframes blink{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.3;transform:scale(.5)}}
  .stxt.running{color:var(--accent3)}
  .stxt.done{color:var(--accent2)}
  .stxt.error{color:#ff4444}

  /* Progress */
  .progress-steps{display:none;flex-wrap:wrap;gap:6px;margin-top:20px}
  .progress-steps.visible{display:flex}
  .step{font-family:var(--mono);font-size:10px;padding:5px 10px;border-radius:3px;
    border:1px solid var(--border);color:var(--text-dim);transition:all .3s}
  .step.active{border-color:var(--accent3);color:var(--accent3);
    background:rgba(255,107,53,.08);box-shadow:var(--glow-orange)}
  .step.done{border-color:var(--accent2);color:var(--accent2);background:rgba(0,255,157,.06)}

  /* Log */
  .log-box{display:none;margin-top:16px;background:rgba(0,0,0,.55);
    border:1px solid var(--border);border-radius:6px;padding:12px 14px;
    font-family:var(--mono);font-size:11px;color:#5a8faa;
    max-height:150px;overflow-y:auto;line-height:1.65;white-space:pre-wrap}
  .log-box.visible{display:block}

  /* Results */
  .results-section{display:none}
  .results-section.visible{display:block}
  .section-title{font-family:var(--display);font-size:11px;font-weight:600;
    color:var(--accent);text-transform:uppercase;letter-spacing:3px;
    margin-bottom:20px;display:flex;align-items:center;gap:10px}
  .section-title::before{content:'';width:16px;height:2px;background:var(--accent);box-shadow:var(--glow)}

  .comparison{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:28px}
  @media(max-width:860px){.comparison{grid-template-columns:1fr}}

  .img-card{background:var(--surface);border:1px solid var(--border);
    border-radius:12px;overflow:hidden;transition:border-color .25s,box-shadow .25s}
  .img-card:hover{border-color:var(--accent);box-shadow:var(--glow)}
  .img-card-hdr{padding:13px 18px;background:var(--surface2);
    border-bottom:1px solid var(--border);display:flex;align-items:center;gap:10px}
  .tag{font-family:var(--mono);font-size:10px;padding:3px 9px;border-radius:3px;
    text-transform:uppercase;letter-spacing:1px}
  .ti{background:rgba(0,212,255,.1);color:var(--accent);border:1px solid rgba(0,212,255,.3)}
  .to{background:rgba(0,255,157,.1);color:var(--accent2);border:1px solid rgba(0,255,157,.3)}
  .card-lbl{font-size:13px;font-weight:600;color:var(--text)}
  .img-card img{display:block;width:100%;object-fit:contain;max-height:420px;
    background:#020408;cursor:zoom-in}

  .pipe-card{background:var(--surface);border:1px solid var(--border);
    border-radius:12px;overflow:hidden;margin-bottom:28px}
  .pipe-hdr{padding:13px 18px;background:var(--surface2);
    border-bottom:1px solid var(--border);font-size:13px;font-weight:600;color:var(--text)}
  .pipe-card img{display:block;width:100%;background:#020408;cursor:zoom-in}

  /* Lightbox */
  .lightbox{display:none;position:fixed;inset:0;z-index:1000;
    background:rgba(0,0,0,.93);align-items:center;justify-content:center}
  .lightbox.open{display:flex}
  .lightbox img{max-width:92vw;max-height:90vh;border:1px solid var(--border);border-radius:8px}
  .lb-x{position:fixed;top:20px;right:28px;color:var(--text);font-size:26px;
    cursor:pointer;font-family:var(--mono);transition:color .2s;z-index:1001}
  .lb-x:hover{color:var(--accent)}

  /* Empty */
  .empty-state{text-align:center;padding:64px 20px;color:var(--text-dim)}
  .empty-state .ei{font-size:50px;margin-bottom:16px;opacity:.3}
  .empty-state p{font-family:var(--mono);font-size:13px}

  ::-webkit-scrollbar{width:5px}
  ::-webkit-scrollbar-track{background:var(--bg)}
  ::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
  ::-webkit-scrollbar-thumb:hover{background:var(--accent)}

  /* Hyperparameters panel */
  .hyper-toggle{display:flex;align-items:center;gap:12px;margin-bottom:0;cursor:pointer;user-select:none}
  .hyper-toggle-label{font-family:var(--mono);font-size:12px;color:var(--text-dim);letter-spacing:1px;text-transform:uppercase}
  .toggle-switch{position:relative;width:42px;height:22px;flex-shrink:0}
  .toggle-switch input{opacity:0;width:0;height:0}
  .toggle-track{position:absolute;inset:0;background:var(--border);border-radius:11px;transition:background .2s;cursor:pointer}
  .toggle-track::before{content:'';position:absolute;height:16px;width:16px;left:3px;top:3px;
    background:var(--text-dim);border-radius:50%;transition:transform .2s,background .2s}
  .toggle-switch input:checked + .toggle-track{background:rgba(0,212,255,.25);border:1px solid var(--accent)}
  .toggle-switch input:checked + .toggle-track::before{transform:translateX(20px);background:var(--accent)}
  .hyper-badge{font-family:var(--mono);font-size:10px;padding:2px 8px;border-radius:3px;
    background:rgba(0,255,157,.08);border:1px solid rgba(0,255,157,.3);color:var(--accent2);letter-spacing:1px}

  .hyper-fields{display:none;margin-top:20px;display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:16px}
  .hyper-fields.hidden{display:none}
  .field-group{display:flex;flex-direction:column;gap:7px}
  .field-label{font-family:var(--mono);font-size:10px;color:var(--text-dim);letter-spacing:1.5px;text-transform:uppercase}
  .field-label span{color:var(--accent2);margin-left:6px;font-size:9px}
  .field-input{background:var(--bg);border:1px solid var(--border);border-radius:5px;
    color:var(--text);font-family:var(--mono);font-size:13px;padding:9px 12px;
    transition:border-color .2s,box-shadow .2s;outline:none;width:100%}
  .field-input:focus{border-color:var(--accent);box-shadow:0 0 8px rgba(0,212,255,.2)}
  .field-input:hover:not(:focus){border-color:rgba(0,212,255,.4)}
  select.field-input{cursor:pointer;appearance:none;
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%234a7a9b'/%3E%3C/svg%3E");
    background-repeat:no-repeat;background-position:right 12px center;padding-right:32px}
  .field-hint{font-family:var(--mono);font-size:10px;color:var(--text-dim);opacity:.7}
  .hyper-divider{border:none;border-top:1px solid var(--border);margin:20px 0 0}
</style>
</head>
<body>

<header>
  <div class="logo-icon">GNR</div>
  <div class="header-text">
    <h1>Building Extraction Pipeline</h1>
    <div class="header-sub">NDBI · CANNY EDGE DETECTION · MORPHOLOGY · HOUGH TRANSFORM</div>
  </div>
  <div class="badge">● SYSTEM ONLINE</div>
</header>

<main>

  <!-- Control Panel -->
  <div class="panel">
    <div class="corner tl"></div><div class="corner tr"></div>
    <div class="corner bl"></div><div class="corner br"></div>
    <div class="panel-title">Input Configuration</div>

    <!-- Mode tabs -->
    <div class="mode-tabs">
      <button class="mode-tab active" id="tabUpload" onclick="setMode('upload')">
        📂 &nbsp;Upload Image from Device
      </button>
      <button class="mode-tab" id="tabSentinel" onclick="setMode('sentinel')">
        🛰️ &nbsp;Sentinel-2 SAT Data
      </button>
    </div>

    <!-- Upload pane -->
    <div id="paneUpload">
      <div class="upload-zone" id="dropZone">
        <input type="file" id="fileInput"
               accept=".png,.jpg,.jpeg,.bmp,.webp,.tif,.tiff"
               onchange="handleFile(this.files[0])"/>
        <div class="up-icon">⬆️</div>
        <div class="up-title">Drop Image Here or Click to Browse</div>
        <div class="up-hint">
          Supports <em>PNG · JPG · BMP · WebP · TIFF</em> &nbsp;—&nbsp; max 256 MB
        </div>
      </div>

      <div class="file-preview" id="filePreview">
        <img class="prev-thumb" id="prevThumb" src="" alt="preview"/>
        <div class="prev-info">
          <div class="prev-name" id="prevName">—</div>
          <div class="prev-meta" id="prevMeta">—</div>
        </div>
        <button class="prev-clear" onclick="clearFile()">✕ Remove</button>
      </div>
    </div>

    <!-- Sentinel pane -->
    <div id="paneSentinel" style="display:none">
      <div class="sentinel-info">
        <div class="sat-icon">🌍</div>
        <div class="sat-det">
          <h3>Sentinel-2 Multispectral Pipeline</h3>
          <p>
            Loads bands B02, B03, B04, B08 &amp; B11, builds an RGB composite,
            and computes a <em>true NDBI</em> from SWIR (B11) and NIR (B08)
            before running edge detection.
          </p>
          <div class="band-pills">
            <span class="band-pill">B02 Blue</span>
            <span class="band-pill">B03 Green</span>
            <span class="band-pill">B04 Red</span>
            <span class="band-pill">B08 NIR</span>
            <span class="band-pill">B11 SWIR</span>
          </div>
        </div>
      </div>

      <!-- Sub-mode: sample on disk vs upload your own -->
      <div class="mode-tabs" style="margin-top:16px">
        <button class="mode-tab active" id="tabSentSample" onclick="setSentinelMode('sample')">
          📁 &nbsp;Use sample data (sat_image/)
        </button>
        <button class="mode-tab" id="tabSentUpload" onclick="setSentinelMode('upload')">
          📤 &nbsp;Upload my own bands
        </button>
      </div>

      <!-- Sample data sub-pane -->
      <div id="paneSentSample" style="margin-top:14px">
        <div class="up-hint" style="text-align:center;padding:18px;border:1px dashed var(--border);border-radius:8px">
          Will run on the TIFFs already present in&nbsp;<em>./sat_image/</em>.
          Click <strong>Execute Pipeline</strong> below.
        </div>
      </div>

      <!-- Upload sub-pane -->
      <div id="paneSentUpload" style="display:none;margin-top:14px">
        <div class="upload-zone" id="sentDropZone">
          <input type="file" id="sentFileInput"
                 accept=".tif,.tiff,.zip" multiple
                 onchange="handleSentFiles(this.files)"/>
          <div class="up-icon">🛰️</div>
          <div class="up-title">Drop your Sentinel-2 TIFFs (or a single .zip)</div>
          <div class="up-hint">
            Required: <em>B08 NIR</em> + <em>B11 SWIR</em> &nbsp;·&nbsp;
            plus <em>True_color</em> OR <em>B02 + B03 + B04</em> for the RGB composite
            &nbsp;—&nbsp; max 256 MB total
          </div>
        </div>

        <div class="file-preview" id="sentFilePreview" style="display:none;flex-direction:column;align-items:flex-start">
          <div class="prev-name" id="sentFileSummary" style="margin-bottom:6px">—</div>
          <div class="prev-meta" id="sentFileList" style="font-size:11px;line-height:1.5">—</div>
          <button class="prev-clear" onclick="clearSentFiles()" style="margin-top:8px;align-self:flex-end">✕ Clear</button>
        </div>
      </div>
    </div>

    <!-- Hyperparameters -->
    <hr class="hyper-divider"/>
    <div style="margin-top:20px">
      <label class="hyper-toggle">
        <span class="toggle-switch">
          <input type="checkbox" id="hyperToggle" onchange="toggleHyper(this.checked)"/>
          <span class="toggle-track"></span>
        </span>
        <span class="hyper-toggle-label">Custom Hyperparameters</span>
        <span class="hyper-badge" id="defaultBadge">DEFAULTS</span>
      </label>

      <div class="hyper-fields hidden" id="hyperFields">
        <div class="field-group">
          <div class="field-label">NDBI Threshold <span>default: 0.1</span></div>
          <input class="field-input" id="ndbiThreshold" type="number" min="-1" max="1" step="0.05"
                 placeholder="0.1"/>
          <div class="field-hint">Range −1 → 1 · higher = stricter built-up filter</div>
        </div>
        <!-- Gaussian -->
        <div class="field-group">
          <div class="field-label">Gaussian Kernel Size <span>default: 5</span></div>
          <input class="field-input" id="gaussKsize" type="number" min="3" max="21" step="2"
                 placeholder="5 (odd integer)" oninput="clampOdd(this)"/>
          <div class="field-hint">Must be odd — e.g. 3, 5, 7, 9 …</div>
        </div>
        <div class="field-group">
          <div class="field-label">Gaussian Sigma (σ) <span>default: 1.4</span></div>
          <input class="field-input" id="gaussSigma" type="number" min="0.3" max="10" step="0.1"
                 placeholder="1.4"/>
          <div class="field-hint">Blur strength — higher = smoother edges</div>
        </div>
        <!-- Morphology -->
        <div class="field-group">
          <div class="field-label">Structuring Element Type <span>default: square</span></div>
          <select class="field-input" id="morphShape">
            <option value="">— use default (square) —</option>
            <option value="square">Square</option>
            <option value="cross">Cross</option>
          </select>
          <div class="field-hint">Shape of the morphological kernel</div>
        </div>
        <div class="field-group">
          <div class="field-label">Structuring Element Size <span>default: 3</span></div>
          <input class="field-input" id="morphSize" type="number" min="3" max="15" step="2"
                 placeholder="3 (odd integer)" oninput="clampOdd(this)"/>
          <div class="field-hint">Must be odd — e.g. 3, 5, 7 …</div>
        </div>
      </div>
    </div>

    <!-- Run row -->
    <div class="run-row">
      <button class="run-btn" id="runBtn" onclick="startRun()" disabled>
        ▶ EXECUTE PIPELINE
      </button>
      <div class="status-bar">
        <div class="sdot" id="sdot"></div>
        <div class="stxt" id="stxt">Upload an image or switch to Sentinel-2 mode, then click Execute</div>
      </div>
    </div>

    <!-- Progress steps -->
    <div class="progress-steps" id="progressSteps">
      <div class="step" id="step1">1/6 Read Image</div>
      <div class="step" id="step2">2/6 Compute NDBI</div>
      <div class="step" id="step3">3/6 Suppress Non-Built</div>
      <div class="step" id="step4">4/6 Canny Edges</div>
      <div class="step" id="step5">5/6 Morphology</div>
      <div class="step" id="step6">6/6 Hough Transform</div>
    </div>

    <!-- Log -->
    <div class="log-box" id="logBox"></div>
  </div>

  <!-- Results -->
  <div class="results-section" id="results">
    <div class="section-title">Input vs Output</div>
    <div class="comparison">
      <div class="img-card">
        <div class="img-card-hdr">
          <span class="tag ti">INPUT</span>
          <span class="card-lbl" id="inLabel">Original Image</span>
        </div>
        <img id="inImg" src="" alt="Input" onclick="lb(this.src)"/>
      </div>
      <div class="img-card">
        <div class="img-card-hdr">
          <span class="tag to">OUTPUT</span>
          <span class="card-lbl">Detected Building Outlines</span>
        </div>
        <img id="outImg" src="" alt="Output" onclick="lb(this.src)"/>
      </div>
    </div>

    <div class="section-title">Full Pipeline Overview</div>
    <div class="pipe-card">
      <div class="pipe-hdr">8-Panel Diagnostic Summary &mdash; click to zoom</div>
      <img id="pipeImg" src="" alt="Pipeline" onclick="lb(this.src)"/>
    </div>
  </div>

  <!-- Empty state -->
  <div id="emptyState" class="empty-state">
    <div class="ei">🛰️</div>
    <p>Upload an image from your device and execute the pipeline to see results</p>
  </div>

</main>

<!-- Lightbox -->
<div class="lightbox" id="lightbox" onclick="closeLb()">
  <div class="lb-x" onclick="closeLb()">✕</div>
  <img id="lbImg" src="" alt=""/>
</div>

<script>
let mode        = 'upload';       // 'upload' | 'sentinel'
let sentMode    = 'sample';       // 'sample' | 'upload' (sub-mode within sentinel)
let chosenFile  = null;
let sentFiles   = [];             // FileList for Sentinel multi-upload
let pollTimer   = null;
let currentJob  = null;

/* ── Hyperparameter helpers ───────────────────────────────────────────── */
function toggleHyper(on) {
  document.getElementById('hyperFields').classList.toggle('hidden', !on);
  document.getElementById('defaultBadge').textContent = on ? 'CUSTOM' : 'DEFAULTS';
  document.getElementById('defaultBadge').style.background    = on ? 'rgba(255,107,53,.08)' : '';
  document.getElementById('defaultBadge').style.borderColor   = on ? 'rgba(255,107,53,.4)'  : '';
  document.getElementById('defaultBadge').style.color         = on ? 'var(--accent3)'        : '';
}

function clampOdd(el) {
  let v = parseInt(el.value, 10);
  if (!isNaN(v) && v % 2 === 0) el.value = v + 1;
}

function collectHyperParams() {
  if (!document.getElementById('hyperToggle').checked) return {};
  const p = {};
  const ksize = document.getElementById('gaussKsize').value.trim();
  const sigma = document.getElementById('gaussSigma').value.trim();
  const shape = document.getElementById('morphShape').value.trim();
  const msize = document.getElementById('morphSize').value.trim();
  const ndbi  = document.getElementById('ndbiThreshold').value.trim();
  if (ksize) p.gauss_ksize    = ksize;
  if (sigma) p.gauss_sigma    = sigma;
  if (shape) p.morph_shape    = shape;
  if (msize) p.morph_size     = msize;
  if (ndbi)  p.ndbi_threshold = ndbi;
  return p;
}

/* ── Mode ─────────────────────────────────────────────────────────────── */
function setMode(m) {
  mode = m;
  document.getElementById('tabUpload').classList.toggle('active',   m==='upload');
  document.getElementById('tabSentinel').classList.toggle('active', m==='sentinel');
  document.getElementById('paneUpload').style.display   = m==='upload'   ? '' : 'none';
  document.getElementById('paneSentinel').style.display = m==='sentinel' ? '' : 'none';
  refreshRunButton();
}

function setSentinelMode(sm) {
  sentMode = sm;
  document.getElementById('tabSentSample').classList.toggle('active', sm==='sample');
  document.getElementById('tabSentUpload').classList.toggle('active', sm==='upload');
  document.getElementById('paneSentSample').style.display = sm==='sample' ? '' : 'none';
  document.getElementById('paneSentUpload').style.display = sm==='upload' ? '' : 'none';
  refreshRunButton();
}

function refreshRunButton() {
  if (mode === 'upload') {
    chosenFile ? enableRun('Ready: '+chosenFile.name)
               : disableRun('Upload an image to continue');
  } else {
    if (sentMode === 'sample') {
      enableRun('Ready — will use sat_image/ folder');
    } else {
      sentFiles.length
        ? enableRun('Ready: '+sentFiles.length+' file'+(sentFiles.length>1?'s':'')+' to upload')
        : disableRun('Drop your Sentinel-2 TIFFs (or a .zip) to continue');
    }
  }
}

function enableRun(msg)  { document.getElementById('runBtn').disabled=false; setStatus('',msg); }
function disableRun(msg) { document.getElementById('runBtn').disabled=true;  setStatus('',msg); }

/* ── Drag & drop ───────────────────────────────────────────────────────── */
const dz = document.getElementById('dropZone');
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag-over'); });
dz.addEventListener('dragleave', ()=> dz.classList.remove('drag-over'));
dz.addEventListener('drop', e => {
  e.preventDefault(); dz.classList.remove('drag-over');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

const sdz = document.getElementById('sentDropZone');
if (sdz) {
  sdz.addEventListener('dragover',  e => { e.preventDefault(); sdz.classList.add('drag-over'); });
  sdz.addEventListener('dragleave', ()=> sdz.classList.remove('drag-over'));
  sdz.addEventListener('drop', e => {
    e.preventDefault(); sdz.classList.remove('drag-over');
    if (e.dataTransfer.files.length) handleSentFiles(e.dataTransfer.files);
  });
}

/* ── File chosen (upload mode) ─────────────────────────────────────────── */
function handleFile(file) {
  if (!file) return;
  chosenFile = file;
  const reader = new FileReader();
  reader.onload = ev => { document.getElementById('prevThumb').src = ev.target.result; };
  reader.readAsDataURL(file);
  document.getElementById('prevName').textContent = file.name;
  document.getElementById('prevMeta').textContent =
    (file.size/1024/1024).toFixed(2)+' MB  ·  '+(file.type||'image');
  document.getElementById('filePreview').classList.add('visible');
  enableRun('Ready: '+file.name);
}

function clearFile() {
  chosenFile = null;
  document.getElementById('fileInput').value = '';
  document.getElementById('prevThumb').src   = '';
  document.getElementById('filePreview').classList.remove('visible');
  disableRun('Upload an image to continue');
}

/* ── Sentinel multi-file handler ───────────────────────────────────────── */
function handleSentFiles(fileList) {
  // Filter to only the supported extensions and warn about any rejects
  const accepted = [];
  const rejected = [];
  for (const f of fileList) {
    const ext = f.name.toLowerCase().match(/\.(tif|tiff|zip)$/);
    if (ext) accepted.push(f); else rejected.push(f.name);
  }
  if (!accepted.length) {
    setStatus('error', 'No .tif/.tiff/.zip files in selection');
    return;
  }
  sentFiles = accepted;

  const totalMB = accepted.reduce((s,f)=>s+f.size,0) / (1024*1024);
  const detected = detectBands(accepted);
  const missing  = ['B08','B11'].filter(b => !detected.has(b));
  // RGB requirement: True_color or all of B02/B03/B04
  const hasTC = accepted.some(f => /true_color/i.test(f.name));
  if (!hasTC) {
    for (const b of ['B02','B03','B04']) if (!detected.has(b)) missing.push(b);
  }

  const summary = accepted.length + ' file' + (accepted.length>1?'s':'') +
                  '  ·  ' + totalMB.toFixed(2) + ' MB' +
                  (rejected.length ? '  ·  '+rejected.length+' rejected' : '');
  document.getElementById('sentFileSummary').innerHTML = summary +
    (missing.length
       ? ' &nbsp;<span style="color:var(--accent3)">⚠ missing: '+missing.join(', ')+'</span>'
       : ' &nbsp;<span style="color:var(--accent2)">✓ all required bands present</span>');

  document.getElementById('sentFileList').innerHTML =
    accepted.map(f => '• ' + f.name + '  <span style="color:var(--text-dim)">('+ (f.size/1024/1024).toFixed(2) +' MB)</span>').join('<br/>');
  document.getElementById('sentFilePreview').style.display = 'flex';

  refreshRunButton();
}

function clearSentFiles() {
  sentFiles = [];
  document.getElementById('sentFileInput').value = '';
  document.getElementById('sentFilePreview').style.display = 'none';
  refreshRunButton();
}

function detectBands(files) {
  const bands = new Set();
  const re = /(B0?2|B0?3|B0?4|B0?8|B11)/i;
  for (const f of files) {
    const m = f.name.match(re);
    if (m) {
      // Normalise to uppercase, two-digit form
      let b = m[1].toUpperCase();
      if (b.length === 2) b = 'B0' + b[1];   // 'B2' -> 'B02'
      bands.add(b);
    }
  }
  return bands;
}

/* ── Status helpers ────────────────────────────────────────────────────── */
function setStatus(cls, txt) {
  const dot = document.getElementById('sdot');
  const tx  = document.getElementById('stxt');
  dot.className = 'sdot'+(cls?' '+cls:'');
  tx.className  = 'stxt'+(cls?' '+cls:'');
  tx.textContent = txt;
}

function updateSteps(log) {
  const m = ['[1/','[2/','[3/','[4/','[5/','[6/'];
  let last = -1;
  m.forEach((s,i)=>{ if(log.includes(s)) last=i; });
  for(let i=0;i<m.length;i++) {
    const el = document.getElementById('step'+(i+1));
    el.className = i<last ? 'step done' : i===last ? 'step active' : 'step';
  }
}

/* ── Run ───────────────────────────────────────────────────────────────── */
async function startRun() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer=null; }

  document.getElementById('runBtn').disabled=true;
  document.getElementById('logBox').classList.add('visible');
  document.getElementById('logBox').textContent='';
  document.getElementById('progressSteps').classList.add('visible');
  document.getElementById('results').classList.remove('visible');
  document.getElementById('emptyState').style.display='none';
  for(let i=1;i<=6;i++) document.getElementById('step'+i).className='step';

  setStatus('running','Uploading and submitting job…');

  try {
    let resp, data;
    const hyper = collectHyperParams();

    if (mode === 'sentinel' && sentMode === 'upload') {
      if (!sentFiles.length) throw new Error('No Sentinel-2 files selected');
      const fd = new FormData();
      for (const f of sentFiles) fd.append('bands', f);
      Object.entries(hyper).forEach(([k,v]) => fd.append(k, v));
      resp = await fetch('/run_sentinel_upload', {method:'POST', body:fd});
      data = await resp.json();

    } else if (mode === 'sentinel') {
      resp = await fetch('/run_sentinel', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify(hyper)
      });
      data = await resp.json();

    } else {
      if (!chosenFile) throw new Error('No file selected');
      const fd = new FormData();
      fd.append('image', chosenFile);
      Object.entries(hyper).forEach(([k,v]) => fd.append(k, v));
      resp = await fetch('/run_upload', {method:'POST', body:fd});
      data = await resp.json();
    }

    if (!resp.ok || data.error) throw new Error(data.error || 'Server error');
    currentJob = data.job_id;
    setStatus('running','Running pipeline: '+data.label);
    pollTimer = setInterval(()=>pollJob(currentJob), 1200);
  } catch(err) {
    setStatus('error','Error: '+err.message);
    document.getElementById('runBtn').disabled=false;
  }
}

/* ── Poll ──────────────────────────────────────────────────────────────── */
async function pollJob(jobId) {
  try {
    const resp = await fetch('/poll/'+jobId);
    const job  = await resp.json();
    if (job.log) {
      const box=document.getElementById('logBox');
      box.textContent=job.log; box.scrollTop=box.scrollHeight;
      updateSteps(job.log);
    }
    if (job.status==='done') {
      clearInterval(pollTimer); pollTimer=null;
      setStatus('done','✓ Pipeline complete — results ready');
      for(let i=1;i<=6;i++) document.getElementById('step'+i).className='step done';
      document.getElementById('inLabel').textContent = job.label||'Input Image';
      document.getElementById('inImg').src    = job.input_img    ||'';
      document.getElementById('outImg').src   = job.output_img   ||'';
      document.getElementById('pipeImg').src  = job.pipeline_img ||'';
      document.getElementById('results').classList.add('visible');
      document.getElementById('runBtn').disabled=false;
    } else if (job.status==='error') {
      clearInterval(pollTimer); pollTimer=null;
      setStatus('error','Error: '+(job.error||'unknown'));
      document.getElementById('runBtn').disabled=false;
    }
  } catch(e){ console.error(e); }
}

/* ── Lightbox ──────────────────────────────────────────────────────────── */
function lb(src) { document.getElementById('lbImg').src=src; document.getElementById('lightbox').classList.add('open'); }
function closeLb(){ document.getElementById('lightbox').classList.remove('open'); }
document.addEventListener('keydown',e=>{ if(e.key==='Escape') closeLb(); });
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("=" * 60)
    print("  GNR Building Extraction — Web UI")
    print("  Open: http://localhost:5000")
    print("=" * 60)
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
