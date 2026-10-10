import os, shutil, subprocess, tempfile, threading, time, uuid, secrets, glob
from fastapi import Request, FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import dedup, transcriber

app = FastAPI()
for _d in glob.glob(tempfile.gettempdir() + '/vocr_*'):  # purge leftovers from a crashed run
    shutil.rmtree(_d, ignore_errors=True)
MAX_BYTES = int(os.environ.get("MAX_MB", "500")) * 1024 * 1024
MAX_SECONDS = int(os.environ.get("MAX_SECONDS", "300"))
FPS = float(os.environ.get("FPS", "3"))
JOBS = {}
import logging, sys
log = logging.getLogger('vocr'); log.setLevel(logging.INFO)
_h = logging.FileHandler('/tmp/jobs.log'); _h.setFormatter(logging.Formatter('%(asctime)s %(message)s')); log.addHandler(_h)
RUNTIME = {"key": os.environ.get("GEMINI_API_KEY", "")}
SETUP_TOKEN = os.environ.get("SETUP_TOKEN", "")
sem = threading.Semaphore(int(os.environ.get("CONCURRENCY", "1")))


def update(job, **kw):
    job.update(kw)


def run(job_id, path, workdir):
    job = JOBS[job_id]
    try:
        with sem:
            _run(job, path, workdir)
    except Exception as e:
        update(job, status="error", error=str(e)[:300])
        log.info("job=%s ERROR %s | %s", job_id, str(e)[:500], {k: job.get(k) for k in ("duration","frames_total","strips","chunks_done","chunks_total")})
    finally:
        shutil.rmtree(workdir, ignore_errors=True)  # video + frames discarded


def _run(job, path, workdir):
    update(job, status="probing", stage="Reading video")
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    try:
        dur = float(p.stdout.strip())
    except ValueError:
        raise RuntimeError("Could not read this video file")
    if dur > MAX_SECONDS:
        raise RuntimeError(f"Video is {int(dur)}s; this demo allows up to {MAX_SECONDS}s")
    # keep long videos tractable: fewer samples per second as duration grows
    update(job, duration=dur, stage="Extracting frames", fps=min(FPS, max(1.0, 700.0 / dur)))
    fdir = os.path.join(workdir, "f"); os.makedirs(fdir)
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{1/job['fps']:.4f})',scale=720:-2", "-vsync", "vfr", "-q:v", "3",
                        os.path.join(fdir, "%05d.jpg")], capture_output=True, text=True)
    files = sorted(glob.glob(fdir + "/*.jpg"))
    if not files:
        raise RuntimeError("No frames could be extracted")
    os.remove(path)
    update(job, status="dedup", stage="Removing duplicate frames", frames_total=len(files))
    def lazy():  # load one frame at a time so long videos never exhaust RAM
        for i, f in enumerate(files):
            yield i / job['fps'], Image.open(f).convert("RGB")
            if i % 20 == 0:
                update(job, stage=f"Analyzing frames {i}/{len(files)}")
    strips, st = dedup.build_strips(lazy())
    first = Image.open(files[0]); all_px = first.height * len(files); first.close()
    sent_px = sum(s[1].height for s in strips)
    update(job, frames_total=st["frames_total"], strips=len(strips), dedup_rate=round(100 * (1 - sent_px / all_px), 1))
    update(job, status="transcribing", stage="Transcribing", chunks_done=0, chunks_total=0)
    key = RUNTIME["key"]
    if not key:
        raise RuntimeError("Server API key not configured")
    def prog(n, tot):
        update(job, chunks_done=n, chunks_total=tot)
    text, tokens = transcriber.transcribe(strips, key, prog)
    update(job, status="done", stage="Done", text=text, tokens=tokens)
    log.info("job=%s DONE dur=%.0fs frames=%s strips=%s dedup=%s chunks=%s tokens=%s chars=%d elapsed=%.0fs", job["id"] if "id" in job else "?", job.get("duration", 0), job.get("frames_total"), job.get("strips"), job.get("dedup_rate"), job.get("chunks_total"), tokens, len(text), time.time() - job["created"])


@app.post("/api/jobs")
async def create(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in (".mp4", ".mov", ".webm", ".m4v", ".mkv"):
        raise HTTPException(400, "Use .mp4, .mov or .webm")
    workdir = tempfile.mkdtemp(prefix="vocr_")
    path = os.path.join(workdir, "in" + ext)
    size = 0
    with open(path, "wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_BYTES:
                f.close(); shutil.rmtree(workdir, ignore_errors=True)
                raise HTTPException(413, f"File too large (max {MAX_BYTES >> 20} MB)")
            f.write(chunk)
    jid = uuid.uuid4().hex[:12]
    log.info("job=%s UPLOAD name_ext=%s size=%.1fMB", jid, ext, size / 1e6)
    JOBS[jid] = {"id": jid, "status": "queued", "stage": "Queued", "created": time.time()}
    threading.Thread(target=run, args=(jid, path, workdir), daemon=True).start()
    # purge old jobs (no persistence: results live in memory ~30 min)
    for k in [k for k, v in JOBS.items() if time.time() - v["created"] > 1800]:
        JOBS.pop(k, None)
    return {"id": jid}


UPLOADS = {}


@app.post("/api/uploads")
def up_init(name: str = Form(...), size: int = Form(...)):
    ext = os.path.splitext(name)[1].lower()
    if ext not in (".mp4", ".mov", ".webm", ".m4v", ".mkv"):
        raise HTTPException(400, "Use .mp4, .mov or .webm")
    if size > MAX_BYTES:
        raise HTTPException(413, f"File too large (max {MAX_BYTES >> 20} MB)")
    uid = uuid.uuid4().hex[:12]
    workdir = tempfile.mkdtemp(prefix="vocr_")
    UPLOADS[uid] = {"dir": workdir, "ext": ext, "size": size, "got": 0, "t": time.time()}
    return {"id": uid}


@app.put("/api/uploads/{uid}/{n}")
async def up_chunk(uid: str, n: int, request: Request):
    u = UPLOADS.get(uid)
    if not u:
        raise HTTPException(404, "Upload expired, please retry")
    data = await request.body()
    p = os.path.join(u["dir"], f"part{n:05d}")
    if os.path.exists(p):
        u["got"] -= os.path.getsize(p)  # retried chunk replaces the old one
    with open(p, "wb") as f:
        f.write(data)
    u["got"] += len(data)
    if u["got"] > MAX_BYTES:
        shutil.rmtree(u["dir"], ignore_errors=True); UPLOADS.pop(uid, None)
        raise HTTPException(413, "File too large")
    return {"ok": True}


@app.post("/api/uploads/{uid}/finish")
def up_finish(uid: str):
    u = UPLOADS.pop(uid, None)
    if not u:
        raise HTTPException(404, "Upload expired, please retry")
    path = os.path.join(u["dir"], "in" + u["ext"])
    with open(path, "wb") as out:
        for part in sorted(glob.glob(u["dir"] + "/part*")):
            with open(part, "rb") as f:
                shutil.copyfileobj(f, out)
            os.remove(part)
    if os.path.getsize(path) != u["size"]:
        shutil.rmtree(u["dir"], ignore_errors=True)
        raise HTTPException(400, "Upload incomplete, please retry")
    jid = uuid.uuid4().hex[:12]
    log.info("job=%s UPLOAD(chunked) ext=%s size=%.1fMB", jid, u["ext"], u["size"] / 1e6)
    JOBS[jid] = {"id": jid, "status": "queued", "stage": "Queued", "created": time.time()}
    threading.Thread(target=run, args=(jid, path, u["dir"]), daemon=True).start()
    return {"id": jid}


@app.get("/api/jobs/{jid}")
def status(jid: str):
    for k in [k for k, v in JOBS.items() if time.time() - v["created"] > 1800]:
        JOBS.pop(k, None)
    for k in [k for k, v in UPLOADS.items() if time.time() - v["t"] > 1800]:
        shutil.rmtree(UPLOADS.pop(k)["dir"], ignore_errors=True)
    j = JOBS.get(jid)
    if not j:
        raise HTTPException(404, "Job expired or not found")
    return {k: v for k, v in j.items() if k != "created"}


@app.get("/api/health")
def health():
    return {"ok": True, "key_configured": bool(RUNTIME["key"])}


@app.get("/setup")
def setup_page():
    if not SETUP_TOKEN:
        raise HTTPException(404)
    return FileResponse("static/setup.html")


@app.post("/api/setup")
def setup(token: str = Form(...), key: str = Form(...)):
    if not SETUP_TOKEN or not secrets.compare_digest(token, SETUP_TOKEN):
        raise HTTPException(403)
    RUNTIME["key"] = key.strip()
    return PlainTextResponse("ok")


@app.middleware("http")
async def nocache(request, call_next):
    resp = await call_next(request)
    if not request.url.path.startswith("/api"):
        resp.headers["Cache-Control"] = "no-cache, must-revalidate"
    return resp


app.mount("/", StaticFiles(directory="static", html=True), name="static")
