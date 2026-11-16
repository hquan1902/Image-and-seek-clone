import sys

print("=" * 60)
print("PYTHON")
print(sys.version)

print("=" * 60)
print("TORCH")

import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU count:", torch.cuda.device_count())

print("=" * 60)
print("CORE IMPORTS (required for the student workflow)")

core_modules = [
    "torchvision",
    "open_clip",
    "PIL",
]

failed = []
for name in core_modules:
    try:
        __import__(name)
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        failed.append(name)

print("=" * 60)
print("LEGACY PROXY STACK (not required; intentionally not installed)")
for name in ("transformers", "diffusers", "cv2", "einops", "shapely", "auto_gptq"):
    try:
        __import__(name)
        print(f"[AVAILABLE] {name}")
    except Exception:
        print(f"[SKIPPED] {name}")

print("=" * 60)
print("CORE STATUS:", "PASS" if not failed else f"FAIL ({', '.join(failed)})")
raise SystemExit(bool(failed))
