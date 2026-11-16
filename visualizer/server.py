"""CIRR demo comparing static image/text fusion and the trained adapter."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "visualizer" / "static"
CAPTIONS = ROOT / "data/CIRR/cirr/captions/cap.rc2.val.json"
SPLIT = ROOT / "data/CIRR/cirr/image_splits/split.rc2.val.json"
BASELINE = ROOT / "runs/student_multidomain_hardneg_seed42/cirr_static.json"
IMPROVED = ROOT / "runs/student_multidomain_group2_ensemble/cirr_val.json"


def load_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Required demo input is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


CAPTION_ROWS = load_json(CAPTIONS)
IMAGE_SPLIT = load_json(SPLIT)
BASELINE_RESULTS = load_json(BASELINE)
IMPROVED_RESULTS = load_json(IMPROVED)
BASELINE_REPORT = load_json(BASELINE.with_suffix(".metrics.json"))
IMPROVED_REPORT = load_json(IMPROVED.with_suffix(".metrics.json"))


def rank_of(items: list[str], target: str) -> int:
    try:
        return items.index(target) + 1
    except ValueError:
        return 51


def image_path(image_id: str) -> Path | None:
    relative = IMAGE_SPLIT.get(image_id)
    if not relative:
        return None
    root = (ROOT / "data/CIRR").resolve()
    candidate = (root / relative.removeprefix("./")).resolve()
    if root not in candidate.parents or not candidate.is_file():
        return None
    return candidate


CASES = []
CASE_BY_ID = {}
for row in CAPTION_ROWS:
    pair_id = str(row["pairid"])
    baseline = BASELINE_RESULTS.get(pair_id, [])
    improved = IMPROVED_RESULTS.get(pair_id, [])
    target = row["target_hard"]
    baseline_rank = rank_of(baseline, target)
    improved_rank = rank_of(improved, target)
    delta = baseline_rank - improved_rank
    status = "improved" if delta > 0 else "regressed" if delta < 0 else "unchanged"
    case = {
        "pair_id": pair_id,
        "caption": row["caption"],
        "reference": row["reference"],
        "target": target,
        "baseline_rank": baseline_rank,
        "improved_rank": improved_rank,
        "rank_gain": delta,
        "status": status,
        "group_id": row.get("img_set", {}).get("id"),
    }
    CASES.append(case)
    CASE_BY_ID[pair_id] = (case, row)


def pct(value: float) -> float:
    return round(100.0 * value, 2)


def metric_rows():
    old = BASELINE_REPORT["metrics_fraction"]
    new = IMPROVED_REPORT["metrics_fraction"]
    keys = ("R@1", "R@5", "R@10", "R@50", "group_R@1", "group_R@2", "group_R@3")
    return [
        {"name": key, "baseline": pct(old[key]), "improved": pct(new[key]), "delta": round(pct(new[key]) - pct(old[key]), 2)}
        for key in keys
    ]


def result_cards(ids: list[str], target: str, group: set[str]):
    return [
        {
            "rank": idx + 1,
            "image_id": image_id,
            "image_url": f"/api/image/{image_id}",
            "is_target": image_id == target,
            "is_group_member": image_id in group,
        }
        for idx, image_id in enumerate(ids[:12])
    ]


SUMMARY = {
    "dataset": "CIRR validation",
    "queries": len(CASES),
    "gallery_images": len(IMAGE_SPLIT),
    "baseline": "Internal static image/text fusion (image weight = 0.2; no proxies)",
    "improved": "Group-aware residual adapter ensemble (seeds 42, 7, 123)",
    "backbone": "OpenCLIP ViT-g/14 + TargetPad",
    "metrics": metric_rows(),
    "case_counts": {name: sum(c["status"] == name for c in CASES) for name in ("improved", "regressed", "unchanged")},
}


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "CIRRDemo/1.0"

    def log_message(self, fmt, *args):
        print(f"[demo] {self.address_string()} - {fmt % args}")

    def send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, cache=False):
        if not path.is_file():
            self.send_error(404, "File not found")
            return
        body = path.read_bytes()
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=86400" if cache else "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        params = parse_qs(parsed.query)

        if path == "/api/summary":
            self.send_json(SUMMARY)
            return
        if path == "/api/cases":
            status = params.get("status", ["improved"])[0]
            search = params.get("q", [""])[0].strip().lower()
            rows = [c for c in CASES if status == "all" or c["status"] == status]
            if search:
                rows = [c for c in rows if search in c["pair_id"] or search in c["caption"].lower()]
            rows.sort(key=lambda c: (-c["rank_gain"], int(c["pair_id"])))
            self.send_json({"cases": rows, "total": len(rows)})
            return

        match = re.fullmatch(r"/api/case/(\d+)", path)
        if match:
            pair_id = match.group(1)
            record = CASE_BY_ID.get(pair_id)
            if not record:
                self.send_json({"error": "Unknown pair ID"}, 404)
                return
            case, row = record
            group = set(row.get("img_set", {}).get("members", []))
            payload = dict(case)
            payload.update({
                "reference_url": f"/api/image/{case['reference']}",
                "target_url": f"/api/image/{case['target']}",
                "baseline_results": result_cards(BASELINE_RESULTS.get(pair_id, []), case["target"], group),
                "improved_results": result_cards(IMPROVED_RESULTS.get(pair_id, []), case["target"], group),
            })
            self.send_json(payload)
            return

        match = re.fullmatch(r"/api/image/([A-Za-z0-9_-]+)", path)
        if match:
            candidate = image_path(match.group(1))
            if candidate:
                self.send_file(candidate, cache=True)
            else:
                self.send_error(404, "Image not found")
            return

        if path in ("/", "/index.html"):
            self.send_file(STATIC / "index.html")
            return
        if path.startswith("/static/"):
            filename = path.removeprefix("/static/")
            if filename in {"app.js", "style.css"}:
                self.send_file(STATIC / filename)
                return
        self.send_error(404, "Not found")


def main():
    parser = argparse.ArgumentParser(description="CIRR baseline-vs-improved visual demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print("=" * 66)
    print(" Imagine-and-Seek research demo")
    print(f" Open in browser: http://{args.host}:{args.port}")
    print(f" Loaded {len(CASES)} CIRR validation queries; Ctrl+C to stop")
    print("=" * 66)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDemo stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
