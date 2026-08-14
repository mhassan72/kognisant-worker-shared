"""
Shared utilities for all serverless GPU workers.

Handles:
- Redis progress publishing
- Firebase Storage upload
- Firestore billing records
- FCM notifications
- HLS transcoding via ffmpeg
"""

import json
import logging
import os
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import firebase_admin
import httpx
import redis
from firebase_admin import credentials, firestore, messaging, storage
from google.cloud.storage import Bucket

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://redis.batch-inference.svc.cluster.local:6379")
FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH", "/secrets/firebase.json")
FIREBASE_STORAGE_BUCKET = os.getenv("FIREBASE_STORAGE_BUCKET", "")
MARGIN_MULTIPLIER = float(os.getenv("MARGIN_MULTIPLIER", "1.8"))

JOB_PREFIX = "job:"
PROGRESS_CHANNEL = "job:progress:{job_id}"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("worker")

# ---------------------------------------------------------------------------
# Init Firebase
# ---------------------------------------------------------------------------
_firebase_creds_b64 = os.getenv("FIREBASE_CREDENTIALS_B64", "")

if _firebase_creds_b64:
    import base64, tempfile as _tf
    _creds_data = base64.b64decode(_firebase_creds_b64)
    _tmp = _tf.NamedTemporaryFile(mode="wb", suffix=".json", delete=False)
    _tmp.write(_creds_data)
    _tmp.close()
    cred = credentials.Certificate(_tmp.name)
    firebase_admin.initialize_app(cred, {"storageBucket": FIREBASE_STORAGE_BUCKET})
    log.info("Firebase initialized from FIREBASE_CREDENTIALS_B64 env var")
elif os.path.exists(FIREBASE_CREDENTIALS_PATH):
    cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
    firebase_admin.initialize_app(cred, {"storageBucket": FIREBASE_STORAGE_BUCKET})
    log.info(f"Firebase initialized from {FIREBASE_CREDENTIALS_PATH}")
else:
    firebase_admin.initialize_app(options={"storageBucket": FIREBASE_STORAGE_BUCKET})
    log.warning("Firebase initialized without credentials — some features may not work")

db = firestore.client()
bucket: Bucket = storage.bucket()

# ---------------------------------------------------------------------------
# Init Redis
# ---------------------------------------------------------------------------
r = redis.from_url(REDIS_URL, decode_responses=True)


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

# GPU rates ($/ms) for live cost calculation — must match batcher's GPU routing
GPU_RATES = {
    "NVIDIA L40S": 0.000000275,
    "NVIDIA A100 80GB PCIe": 0.000000386,
    "NVIDIA A100-80GB": 0.000000386,
    "NVIDIA RTX A5000": 0.000000075,
    "NVIDIA RTX 4090": 0.000000192,
    "NVIDIA A40": 0.000000122,
    "NVIDIA L4": 0.000000108,
}

# Detect GPU type and rate at startup
_GPU_TYPE = os.getenv("RUNPOD_GPU_TYPE", os.getenv("GPU_TYPE", ""))
_RATE_PER_MS = GPU_RATES.get(_GPU_TYPE, 0.000000275)  # default L40S
_CONTAINER_START: float = 0  # set by the worker at startup


def set_container_start(t: float):
    """Call at the beginning of main() to enable elapsed/cost tracking."""
    global _CONTAINER_START
    _CONTAINER_START = t


def _cost_so_far() -> float:
    """Current cost based on elapsed time since container start."""
    if not _CONTAINER_START:
        return 0.0
    elapsed_ms = (time.time() - _CONTAINER_START) * 1000
    return elapsed_ms * _RATE_PER_MS * MARGIN_MULTIPLIER


def _elapsed_ms() -> int:
    if not _CONTAINER_START:
        return 0
    return int((time.time() - _CONTAINER_START) * 1000)


def publish_progress(job_id: str, progress: int, message: str, status: str = "processing", **extra):
    """Publish progress to Redis Pub/Sub and update job hash.

    Automatically includes elapsed_ms, cost_so_far_usd, gpu_type, and rate_per_ms.
    """
    progress = max(0, min(100, int(progress)))

    event = {
        "job_id": job_id,
        "status": status,
        "progress": progress,
        "progress_message": message,
        "elapsed_ms": _elapsed_ms(),
        "cost_so_far_usd": round(_cost_so_far(), 6),
        "gpu_type": _GPU_TYPE,
        "rate_per_ms": _RATE_PER_MS,
        "margin_multiplier": MARGIN_MULTIPLIER,
        **extra,
    }

    channel = PROGRESS_CHANNEL.format(job_id=job_id)
    r.publish(channel, json.dumps(event))

    raw = r.get(f"{JOB_PREFIX}{job_id}")
    if raw:
        data = json.loads(raw)
        data["status"] = status
        data["progress"] = progress
        data["progress_message"] = message
        data["elapsed_ms"] = event["elapsed_ms"]
        data["cost_so_far_usd"] = event["cost_so_far_usd"]
        data.update(extra)
        r.set(f"{JOB_PREFIX}{job_id}", json.dumps(data))


# ---------------------------------------------------------------------------
# Batch Progress Tracker
# ---------------------------------------------------------------------------
class BatchProgress:
    """Tracks progress across a batch of jobs and maps per-step diffusion
    callbacks to smooth 1-100% per job, with stage labels.

    Stages for image generation:
        0-5%    : Queued / Loading model
        5-10%   : Downloading assets
        10-90%  : Generating (diffusion steps → maps linearly within this range)
        90-95%  : Creating variants
        95-99%  : Uploading
        100%    : Complete

    Stages for video generation:
        0-5%    : Queued / Loading model
        5-10%   : Downloading assets
        10-80%  : Generating (diffusion steps)
        80-90%  : Transcoding
        90-95%  : Creating variants
        95-99%  : Uploading
        100%    : Complete
    """

    def __init__(self, job_ids: list[str], total_steps: int = 28, mode: str = "image"):
        self.job_ids = job_ids
        self.total_jobs = len(job_ids)
        self.total_steps = total_steps
        self.mode = mode
        self.completed_jobs = 0

        # Stage ranges depend on mode
        if mode == "video":
            self.gen_start = 10
            self.gen_end = 80
        else:
            self.gen_start = 10
            self.gen_end = 90

    @property
    def batch_progress(self) -> int:
        """Overall batch progress as a percentage (0-100)."""
        if self.total_jobs == 0:
            return 100
        return int((self.completed_jobs / self.total_jobs) * 100)

    def on_queued(self, job_id: str):
        publish_progress(job_id, 0, "Queued",
                         stage="queued", batch_progress=self.batch_progress,
                         batch_total=self.total_jobs, batch_completed=self.completed_jobs)

    def on_model_loaded(self, job_id: str):
        publish_progress(job_id, 5, "Model loaded",
                         stage="model_loaded", batch_progress=self.batch_progress,
                         batch_total=self.total_jobs, batch_completed=self.completed_jobs)

    def on_downloading_assets(self, job_id: str):
        publish_progress(job_id, 7, "Downloading assets",
                         stage="downloading_assets", batch_progress=self.batch_progress,
                         batch_total=self.total_jobs, batch_completed=self.completed_jobs)

    def on_generating_start(self, job_id: str):
        publish_progress(job_id, self.gen_start, "Generating",
                         stage="generating", batch_progress=self.batch_progress,
                         batch_total=self.total_jobs, batch_completed=self.completed_jobs)

    def on_step(self, job_id: str, step: int, total_steps: int | None = None):
        """Called after each diffusion step. Maps step to a smooth progress within
        the generation range (gen_start to gen_end)."""
        steps = total_steps or self.total_steps
        if steps <= 0:
            return
        # Map step (1-based) to progress within gen range
        fraction = step / steps
        progress = int(self.gen_start + fraction * (self.gen_end - self.gen_start))
        progress = max(self.gen_start, min(self.gen_end, progress))

        publish_progress(job_id, progress, f"Generating (step {step}/{steps})",
                         stage="generating", step=step, total_steps=steps,
                         batch_progress=self.batch_progress,
                         batch_total=self.total_jobs, batch_completed=self.completed_jobs)

    def on_transcoding(self, job_id: str):
        publish_progress(job_id, 82, "Transcoding to HLS",
                         stage="transcoding", batch_progress=self.batch_progress,
                         batch_total=self.total_jobs, batch_completed=self.completed_jobs)

    def on_creating_variants(self, job_id: str):
        pct = 91 if self.mode == "image" else 91
        publish_progress(job_id, pct, "Creating image variants",
                         stage="creating_variants", batch_progress=self.batch_progress,
                         batch_total=self.total_jobs, batch_completed=self.completed_jobs)

    def on_uploading(self, job_id: str):
        pct = 95 if self.mode == "image" else 95
        publish_progress(job_id, pct, "Uploading",
                         stage="uploading", batch_progress=self.batch_progress,
                         batch_total=self.total_jobs, batch_completed=self.completed_jobs)

    def on_complete(self, job_id: str):
        self.completed_jobs += 1
        publish_progress(job_id, 100, "Complete", status="completed",
                         stage="complete", batch_progress=self.batch_progress,
                         batch_total=self.total_jobs, batch_completed=self.completed_jobs)

    def on_failed(self, job_id: str, error: str):
        self.completed_jobs += 1
        publish_progress(job_id, 0, f"Failed: {error}", status="failed",
                         stage="failed", error=error,
                         batch_progress=self.batch_progress,
                         batch_total=self.total_jobs, batch_completed=self.completed_jobs)

    def make_step_callback(self, job_id: str):
        """Return a callback function compatible with diffusers' callback_on_step_end.
        Usage: pipe(..., callback_on_step_end=tracker.make_step_callback(job_id))
        """
        def _callback(pipe, step_index, timestep, callback_kwargs):
            # step_index is 0-based, publish as 1-based
            self.on_step(job_id, step_index + 1, self.total_steps)
            return callback_kwargs
        return _callback


def mark_job_completed(job_id: str, result_url: str, hls_url: str):
    """Mark job as completed with result URLs."""
    publish_progress(
        job_id, 100, "Complete",
        status="completed",
        result_url=result_url,
        hls_url=hls_url,
        completed_at=time.time(),
    )

    # Persist generation record to Firestore
    try:
        raw = r.get(f"{JOB_PREFIX}{job_id}")
        if raw:
            job_data = json.loads(raw)
            generation_doc = {
                "user_id": job_data.get("user_id", "unknown"),
                "job_id": job_id,
                "model": job_data.get("model"),
                "mode": job_data.get("mode", "text_to_video"),
                "prompt": job_data.get("prompt"),
                "inputs": job_data.get("inputs", {}),
                "duration_seconds": job_data.get("duration_seconds"),
                "width": job_data.get("width"),
                "height": job_data.get("height"),
                "result_url": result_url,
                "hls_url": hls_url,
                "status": "completed",
                "created_at": job_data.get("created_at"),
                "completed_at": time.time(),
            }
            db.collection("generations").document(job_id).set(generation_doc)
            log.info(f"Generation record saved to Firestore: {job_id}")
    except Exception as e:
        log.warning(f"Failed to write generation record: {e}")


def mark_job_failed(job_id: str, error: str):
    """Mark job as failed."""
    publish_progress(
        job_id, 0, f"Error: {error}",
        status="failed",
        error=error,
        completed_at=time.time(),
    )

    # Persist failed generation record to Firestore
    try:
        raw = r.get(f"{JOB_PREFIX}{job_id}")
        if raw:
            job_data = json.loads(raw)
            generation_doc = {
                "user_id": job_data.get("user_id", "unknown"),
                "job_id": job_id,
                "model": job_data.get("model"),
                "mode": job_data.get("mode", "text_to_video"),
                "prompt": job_data.get("prompt"),
                "inputs": job_data.get("inputs", {}),
                "duration_seconds": job_data.get("duration_seconds"),
                "width": job_data.get("width"),
                "height": job_data.get("height"),
                "result_url": None,
                "hls_url": None,
                "status": "failed",
                "error": error,
                "created_at": job_data.get("created_at"),
                "completed_at": time.time(),
            }
            db.collection("generations").document(job_id).set(generation_doc)
            log.info(f"Failed generation record saved to Firestore: {job_id}")
    except Exception as e:
        log.warning(f"Failed to write generation record: {e}")


# ---------------------------------------------------------------------------
# HLS Transcoding
# ---------------------------------------------------------------------------
def _hls_common_args(output_dir: str, playlist_path: str) -> list[str]:
    return [
        "-pix_fmt", "yuv420p",
        "-hls_time", "4",
        "-hls_list_size", "0",
        "-hls_segment_filename", os.path.join(output_dir, "segment_%03d.ts"),
        playlist_path,
    ]


def transcode_to_hls(input_path: str, output_dir: str) -> str:
    """Transcode mp4 to HLS segments using ffmpeg. Returns playlist path.

    Uses NVENC hardware encoding when available — we are already paying for a
    GPU, and h264_nvenc is roughly 5-10x faster than libx264 on CPU. Transcode
    is ~23% of pod cost with libx264, so this matters.
    Falls back to libx264 if NVENC is unavailable or fails.
    """
    os.makedirs(output_dir, exist_ok=True)
    playlist_path = os.path.join(output_dir, "playlist.m3u8")
    tail = _hls_common_args(output_dir, playlist_path)

    nvenc_cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-rc", "vbr",
        "-cq", "23",
        *tail,
    ]

    try:
        subprocess.run(nvenc_cmd, check=True, capture_output=True)
        log.info("HLS transcode complete (h264_nvenc)")
        return playlist_path
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        stderr = getattr(e, "stderr", None)
        detail = (stderr or b"").decode(errors="replace")[-500:] if stderr else str(e)
        log.warning(f"NVENC transcode failed, falling back to libx264: {detail}")
        # A partial NVENC run may have left segments behind. Clear them so the
        # fallback encode does not leave stale .ts files to be uploaded.
        for stale in Path(output_dir).glob("segment_*.ts"):
            try:
                stale.unlink()
            except OSError:
                pass

    x264_cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        *tail,
    ]
    subprocess.run(x264_cmd, check=True, capture_output=True)
    log.info("HLS transcode complete (libx264 fallback)")
    return playlist_path


# ---------------------------------------------------------------------------
# Firebase Storage Upload
# ---------------------------------------------------------------------------
def upload_file(local_path: str, remote_path: str) -> str:
    """Upload a file to Firebase Storage and return its public URL."""
    blob = bucket.blob(remote_path)
    blob.upload_from_filename(local_path)
    blob.make_public()
    return blob.public_url


def upload_directory(local_dir: str, remote_prefix: str):
    """Upload all files in a directory to Firebase Storage."""
    for file_path in Path(local_dir).iterdir():
        if file_path.is_file():
            remote_path = f"{remote_prefix}/{file_path.name}"
            upload_file(str(file_path), remote_path)


def get_hls_url(job_id: str) -> str:
    """Construct the public HLS playlist URL."""
    return f"https://storage.googleapis.com/{FIREBASE_STORAGE_BUCKET}/videos/{job_id}/hls/playlist.m3u8"


# ---------------------------------------------------------------------------
# Billing
# ---------------------------------------------------------------------------
def write_billing_records(
    batch_id: str,
    jobs: list[dict],
    container_start: float,
    container_end: float,
    model_name: str,
):
    """Write GPU usage records to Firestore for each job in the batch."""
    total_gpu_seconds = container_end - container_start
    batch_size = len(jobs)

    weights = []
    for job in jobs:
        w = job["duration_seconds"] * (job["width"] * job["height"])
        weights.append(w)
    total_weight = sum(weights)

    for job, weight in zip(jobs, weights):
        share = weight / total_weight if total_weight > 0 else 1.0 / batch_size
        raw_seconds = share * total_gpu_seconds
        billed_seconds = raw_seconds * MARGIN_MULTIPLIER

        doc = {
            "user_id": job["user_id"],
            "job_id": job["job_id"],
            "batch_id": batch_id,
            "model": model_name,
            "gpu_seconds": round(billed_seconds, 2),
            "gpu_seconds_raw": round(raw_seconds, 2),
            "margin_multiplier": MARGIN_MULTIPLIER,
            "duration_requested": job["duration_seconds"],
            "resolution": f"{job['width']}x{job['height']}",
            "batch_size": batch_size,
            "batch_total_gpu_seconds": round(total_gpu_seconds, 2),
            "weight": weight,
            "weight_share_percent": round(share * 100, 1),
            "container_start": datetime.fromtimestamp(container_start, tz=timezone.utc).isoformat(),
            "container_end": datetime.fromtimestamp(container_end, tz=timezone.utc).isoformat(),
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "charged": False,
        }

        db.collection("GPU compute").document(job["job_id"]).set(doc)
        log.info(f"Billing: job {job['job_id']} = {billed_seconds:.1f}s ({share*100:.1f}%)")


# ---------------------------------------------------------------------------
# FCM Notifications
# ---------------------------------------------------------------------------
def notify_user(job: dict, status: str):
    """Send FCM push notification if user provided a token."""
    fcm_token = job.get("fcm_token")
    if not fcm_token:
        return
    try:
        title = "Video Ready!" if status == "completed" else "Job Failed"
        body = ("Your video is ready to view."
                if status == "completed"
                else "Something went wrong with your video generation.")
        message = messaging.Message(
            notification=messaging.Notification(title=title, body=body),
            data={"job_id": job["job_id"], "status": status},
            token=fcm_token,
        )
        messaging.send(message)
        log.info(f"FCM sent to user {job['user_id']}")
    except Exception as e:
        log.warning(f"FCM failed: {e}")


# ---------------------------------------------------------------------------
# Job Enrichment
# ---------------------------------------------------------------------------
def enrich_jobs(jobs: list[dict], model_name: str) -> list[dict]:
    """Pull user_id and fcm_token from Redis for each job."""
    for job in jobs:
        raw = r.get(f"{JOB_PREFIX}{job['job_id']}")
        if raw:
            stored = json.loads(raw)
            job["user_id"] = stored.get("user_id", "unknown")
            job["fcm_token"] = stored.get("fcm_token")
        else:
            job["user_id"] = "unknown"
            job["fcm_token"] = None
        job["model"] = model_name
    return jobs


# ---------------------------------------------------------------------------
# RunPod Self-Termination
# ---------------------------------------------------------------------------
def terminate_runpod_pod():
    """Terminate the current RunPod pod via API to stop billing.
    Call this after all jobs are done or on fatal error.
    """
    pod_id = os.getenv("RUNPOD_POD_ID", "")
    api_key = os.getenv("RUNPOD_API_KEY", "")
    if not pod_id or not api_key:
        log.warning("RUNPOD_POD_ID or RUNPOD_API_KEY not set — cannot self-terminate")
        return
    query = """
    mutation podTerminate($input: PodTerminateInput!) {
        podTerminate(input: $input)
    }
    """
    variables = {"input": {"podId": pod_id}}
    try:
        resp = httpx.post(
            "https://api.runpod.io/graphql",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            json={"query": query, "variables": variables},
            timeout=15,
        )
        log.info(f"RunPod pod {pod_id} terminate response: {resp.status_code}")
    except Exception as e:
        log.error(f"Failed to terminate RunPod pod: {e}")


# ---------------------------------------------------------------------------
# Greedy Worker Loop — Keep processing jobs while queue has work for this model
# ---------------------------------------------------------------------------
GREEDY_MAX_JOBS_PER_DRAIN = int(os.getenv("GREEDY_MAX_JOBS", "20"))
# Seconds to wait for more work before self-terminating.
# Every pod pays this once: 30s costs $0.0116 on A100, $0.0083 on L40S.
# The batcher dispatches within ~1s of a job landing in Redis, so 5s is ample.
GREEDY_IDLE_TIMEOUT = float(os.getenv("GREEDY_IDLE_TIMEOUT", "5"))


def drain_queue_for_model(model_name: str, max_jobs: int = GREEDY_MAX_JOBS_PER_DRAIN) -> list[dict]:
    """Pull up to max_jobs pending jobs for the given model from the Redis queue.

    Only takes jobs that match our model. Jobs for other models are pushed back.
    """
    claimed = []
    returned = []

    # Pop all available jobs and filter
    while len(claimed) < max_jobs:
        job_id = r.rpop("jobs:pending")
        if not job_id:
            break

        raw = r.get(f"{JOB_PREFIX}{job_id}")
        if not raw:
            continue

        job = json.loads(raw)
        if job.get("model") == model_name:
            claimed.append(job)
        else:
            # Not for us — push back to the front of the queue
            returned.append(job_id)

    # Return non-matching jobs back to the queue (push to right = back of queue)
    for job_id in reversed(returned):
        r.lpush("jobs:pending", job_id)

    if claimed:
        log.info(f"Greedy drain: claimed {len(claimed)} jobs for {model_name}, returned {len(returned)} others")
    return claimed


def greedy_worker_loop(model_name: str, process_batch_fn, container_start: float):
    """Keep draining the queue and processing jobs until no more work exists for this model.

    Args:
        model_name: The model this worker handles (e.g. "flux-kontext")
        process_batch_fn: Callable that takes a list of job dicts and processes them.
                          Signature: process_batch_fn(jobs: list[dict]) -> None
        container_start: Timestamp when the container started (for billing)

    After all work is done (or queue is empty for this model), terminates the pod.
    """
    import time as _time

    total_processed = 0
    idle_start = None

    while True:
        jobs = drain_queue_for_model(model_name)

        if jobs:
            idle_start = None
            # Enrich with user info from Redis
            jobs = enrich_jobs(jobs, model_name)
            total_processed += len(jobs)

            log.info(f"Greedy loop: processing {len(jobs)} additional jobs (total so far: {total_processed})")

            try:
                process_batch_fn(jobs)
            except Exception as e:
                log.error(f"Greedy loop batch processing error: {e}")
                # Mark remaining jobs as failed
                for job in jobs:
                    try:
                        mark_job_failed(job["job_id"], f"Worker error: {e}")
                    except Exception:
                        pass

            # Write billing for this mini-batch
            batch_end = _time.time()
            try:
                write_billing_records(
                    str(uuid.uuid4()), jobs, container_start, batch_end, model_name
                )
            except Exception as e:
                log.error(f"Billing write failed: {e}")
            # Update container_start for next billing window
            container_start = batch_end
        else:
            # No jobs found — wait a bit before checking again
            if idle_start is None:
                idle_start = _time.time()
                log.info(f"Greedy loop: queue empty for {model_name}, waiting up to {GREEDY_IDLE_TIMEOUT}s...")

            elapsed_idle = _time.time() - idle_start
            if elapsed_idle >= GREEDY_IDLE_TIMEOUT:
                log.info(f"Greedy loop: idle for {elapsed_idle:.0f}s, no more work. "
                         f"Total processed: {total_processed}. Terminating pod.")
                break

            _time.sleep(2)  # Poll every 2 seconds

    terminate_runpod_pod()


# ---------------------------------------------------------------------------
# Asset Downloads
# ---------------------------------------------------------------------------
def download_asset(url: str, dest_dir: str, filename: str | None = None) -> str:
    """
    Download a file from URL to a local path. Returns the local path.
    Supports Firebase Storage URLs and generic HTTPS URLs.
    """
    if not filename:
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path) or "asset"

    local_path = os.path.join(dest_dir, filename)

    # If it's a gs:// URL, use Firebase Storage SDK
    if url.startswith("gs://"):
        blob_path = url.split("/", 3)[-1] if len(url.split("/")) > 3 else ""
        blob = bucket.blob(blob_path)
        blob.download_to_filename(local_path)
    else:
        # HTTP(S) download
        with httpx.Client(timeout=120, follow_redirects=True) as client:
            with client.stream("GET", url) as resp:
                resp.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)

    log.info(f"Downloaded asset: {url} -> {local_path} ({os.path.getsize(local_path)} bytes)")
    return local_path


def download_assets_for_job(job: dict, work_dir: str) -> dict:
    """
    Download all asset URLs referenced in a job's inputs.
    Returns a dict mapping input field names to local file paths.
    """
    inputs = job.get("inputs", {})
    local_assets = {}

    # Single file fields
    single_fields = [
        ("start_image_url", "start_image"),
        ("end_image_url", "end_image"),
        ("audio_url", "audio"),
        ("source_video_url", "source_video"),
        ("control_video_url", "control_video"),
    ]

    for url_field, local_name in single_fields:
        url = inputs.get(url_field)
        if url:
            ext = os.path.splitext(urlparse(url).path)[1] or ".bin"
            local_path = download_asset(url, work_dir, f"{local_name}{ext}")
            local_assets[url_field] = local_path

    # Multi-file fields
    multi_fields = [
        ("reference_images", "ref_img"),
        ("keyframe_images", "keyframe"),
    ]

    for url_field, prefix in multi_fields:
        urls = inputs.get(url_field, [])
        if urls:
            local_assets[url_field] = []
            for i, url in enumerate(urls):
                ext = os.path.splitext(urlparse(url).path)[1] or ".png"
                local_path = download_asset(url, work_dir, f"{prefix}_{i}{ext}")
                local_assets[url_field].append(local_path)

    return local_assets
