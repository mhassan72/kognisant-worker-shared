# Worker Shared Module

Common utilities used by all GPU worker containers (LTX, Wan, HunyuanVideo).

## Usage

This repo is included as a git submodule in each worker repository:

```bash
git submodule add git@github.com:kognisant/worker-shared.git shared
```

## What's included

- `common.py` — Redis progress, Firebase Storage upload, Firestore billing/generations, FCM notifications, HLS transcoding, asset download utilities

## In Docker

```dockerfile
COPY shared /app/shared
```

Workers reference it via:
```python
sys.path.insert(0, "/app/shared")
from common import log, publish_progress, mark_job_completed, ...
```
