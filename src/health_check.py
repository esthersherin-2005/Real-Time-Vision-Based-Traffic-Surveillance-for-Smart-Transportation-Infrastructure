# health_check.py
"""
Project health-check script for YOLOv8 vehicle_detection project.
Run from the project root with the venv active:
    python health_check.py
"""

import ast
import importlib
import os
import sys
import traceback

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT, "src")
MODELS_DIR = os.path.join(ROOT, "models")
DATASET_DIR = os.path.join(ROOT, "dataset")

FILES_TO_CHECK = []
if os.path.isdir(SRC_DIR):
    for f in os.listdir(SRC_DIR):
        if f.endswith(".py"):
            FILES_TO_CHECK.append(os.path.join(SRC_DIR, f))

def syntax_check(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            src = fh.read()
        ast.parse(src, filename=file_path)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError in {file_path}: {e}"
    except Exception as e:
        return False, f"Error parsing {file_path}: {e}"

def try_import_module(src_dir, module_name):
    try:
        # Ensure src dir is on sys.path
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        m = importlib.import_module(module_name)
        return True, None
    except Exception as e:
        tb = traceback.format_exc()
        return False, tb

def check_models_and_data():
    msgs = []
    if not os.path.isdir(MODELS_DIR):
        msgs.append(f"Models directory missing: {MODELS_DIR}")
    else:
        weights = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pt") or f.endswith(".pth")]
        if not weights:
            msgs.append(f"No .pt/.pth weights found in models/. Add yolov8n.pt or your custom weights.")
        else:
            msgs.append(f"Found weight(s): {weights}")

    if not os.path.isdir(DATASET_DIR):
        msgs.append(f"Dataset directory missing: {DATASET_DIR}")
    else:
        vids = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
        if not vids:
            msgs.append(f"No video files found in dataset/. Add sample_video.mp4 or your input.")
        else:
            msgs.append(f"Found dataset video(s): {vids}")
    return msgs

def main():
    print("PROJECT HEALTH CHECK\nRoot:", ROOT)
    print("\n1) Syntax check for src/*.py")
    ok_all = True
    for f in FILES_TO_CHECK:
        ok, msg = syntax_check(f)
        if ok:
            print("  [OK] ", os.path.basename(f))
        else:
            ok_all = False
            print("  [ERROR]", msg)
    print("\n2) Try importing modules from src/")
    module_files = [os.path.basename(p)[:-3] for p in FILES_TO_CHECK]
    for mod in module_files:
        ok, tb = try_import_module(SRC_DIR, mod)
        if ok:
            print(f"  [OK] import {mod}")
        else:
            ok_all = False
            print(f"  [ERROR] import {mod} failed. Traceback:\n{tb}")

    print("\n3) Models and Dataset checks")
    msgs = check_models_and_data()
    for m in msgs:
        print("  -", m)

    print("\nSUMMARY:")
    if ok_all:
        print("  All quick checks passed. Try running: python src/main.py --source dataset/sample_video.mp4")
    else:
        print("  Some checks failed. See errors above and fix them before running main.py")

if __name__ == "__main__":
    main()
