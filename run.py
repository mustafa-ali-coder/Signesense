"""
run.py -- SignSense single-command launcher.

    python run.py

Does everything in order:
  1. Checks mediapipe and opencv are importable (no auto-install).
  2. Runs the test suite (aborts if anything fails).
  3. Looks at what datasets exist and routes you:
       - no datasets        -> recorder so you can teach letters
       - datasets exist     -> menu: run app / record more / record dynamic

Flags (optional, skip the menu):
    python run.py app                launch live recognition directly
    python run.py record static      record fingerspelling letters
    python run.py record dynamic     record motion signs
    python run.py test               tests only
"""

import importlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_PATH = os.path.join(HERE, "static_signs.json")
DYNAMIC_PATH = os.path.join(HERE, "dynamic_signs.json")


def ensure_deps():
    for module in ("mediapipe", "cv2"):
        try:
            importlib.import_module(module)
        except ImportError:
            print(f"\nMissing: {module}")
            print("Run this once manually, then rerun run.py:")
            print("  pip install opencv-python")
            print("  pip install mediapipe --no-deps")
            print("  pip install absl-py flatbuffers matplotlib numpy")
            return False
    return True


def run_tests():
    print("\n--- running test suite ---")
    result = subprocess.run(
        [sys.executable, os.path.join(HERE, "test_signs.py")],
        cwd=HERE)
    if result.returncode != 0:
        print("\nTests FAILED -- not launching. Fix the above first.")
        return False
    return True


def dataset_summary():
    """Print what's trained so far. Returns (has_static, has_dynamic)."""
    has_static = os.path.exists(STATIC_PATH)
    has_dynamic = os.path.exists(DYNAMIC_PATH)
    print("\n--- datasets ---")
    if has_static:
        from static_sign import StaticSignClassifier
        counts = StaticSignClassifier.load(STATIC_PATH).label_counts()
        print(f"static  : {len(counts)} labels, "
              f"{sum(counts.values())} samples  {sorted(counts)}")
    else:
        print("static  : none")
    if has_dynamic:
        from dynamic_sign import DynamicSignClassifier
        counts = DynamicSignClassifier.load(DYNAMIC_PATH).label_counts()
        print(f"dynamic : {len(counts)} labels, "
              f"{sum(counts.values())} samples  {sorted(counts)}")
    else:
        print("dynamic : none")
    return has_static, has_dynamic


def launch_app():
    import sign_app
    sign_app.main()


def launch_recorder(mode):
    import record_signs
    if mode == "static":
        record_signs.run_static()
    else:
        record_signs.run_dynamic()


def menu(has_static, has_dynamic):
    if not has_static and not has_dynamic:
        print("\nNo datasets yet -- launching the recorder so you can "
              "teach some letters first.")
        print("(Type a label like A, hold the pose, SPACE to capture, "
              "10-15 samples each. Q saves and quits.)\n")
        launch_recorder("static")
        if os.path.exists(STATIC_PATH):
            ans = input("\nLaunch live recognition now? [Y/n] ").strip()
            if ans.lower() != "n":
                launch_app()
        return

    print("\n1) live recognition app"
          "\n2) record more static letters"
          "\n3) record dynamic signs"
          "\n4) quit")
    choice = input("> ").strip()
    if choice == "1":
        launch_app()
    elif choice == "2":
        launch_recorder("static")
    elif choice == "3":
        launch_recorder("dynamic")


def main():
    os.chdir(HERE)
    if HERE not in sys.path:
        sys.path.insert(0, HERE)

    args = [a.lower() for a in sys.argv[1:]]

    if args[:1] == ["test"]:
        sys.exit(0 if run_tests() else 1)

    if not ensure_deps():
        sys.exit(1)
    if not run_tests():
        sys.exit(1)

    if args[:1] == ["app"]:
        launch_app()
        return
    if args[:1] == ["record"]:
        mode = args[1] if len(args) > 1 else "static"
        if mode not in ("static", "dynamic"):
            print("usage: python run.py record [static|dynamic]")
            sys.exit(1)
        launch_recorder(mode)
        return

    has_static, has_dynamic = dataset_summary()
    menu(has_static, has_dynamic)


if __name__ == "__main__":
    main()
