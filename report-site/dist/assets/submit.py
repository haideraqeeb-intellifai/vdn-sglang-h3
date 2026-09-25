#!/usr/bin/env python3
"""Submit one exact VDN-H3 request, poll it, and save the MP4 plus metadata."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent


def read_json(url: str, *, data: bytes | None = None) -> dict:
    request = Request(url, data=data)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urlopen(request, timeout=600) as response:
            return json.load(response)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30010")
    args = parser.parse_args()

    request_payload = json.loads((ROOT / "request.json").read_text())
    request_payload["prompt"] = (ROOT / "prompt.txt").read_text().strip()
    run_dir = ROOT / "runs" / args.label
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "request.json").write_text(
        json.dumps(request_payload, indent=2, ensure_ascii=False) + "\n"
    )

    started = time.time()
    submitted = read_json(
        f"{args.base_url}/v1/videos",
        data=json.dumps(request_payload).encode("utf-8"),
    )
    (run_dir / "submitted.json").write_text(json.dumps(submitted, indent=2) + "\n")
    job_id = submitted["id"]

    while True:
        status = read_json(f"{args.base_url}/v1/videos/{job_id}")
        (run_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        if status["status"] == "completed":
            break
        if status["status"] == "failed":
            raise RuntimeError(json.dumps(status, indent=2))
        time.sleep(2)

    with urlopen(f"{args.base_url}/v1/videos/{job_id}/content", timeout=600) as response:
        (run_dir / "output.mp4").write_bytes(response.read())
    elapsed = time.time() - started
    (run_dir / "client_timing.json").write_text(
        json.dumps({"elapsed_seconds": elapsed, "job_id": job_id}, indent=2) + "\n"
    )
    print(json.dumps({"run_dir": str(run_dir), "elapsed_seconds": elapsed}))


if __name__ == "__main__":
    main()
