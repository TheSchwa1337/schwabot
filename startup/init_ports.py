import importlib
import os

REQUIRED_MODULES = [
    "numpy", "pandas", "matplotlib", "seaborn", "plotly", "psutil", "yaml", "GPUtil", "ta-lib"
]
REQUIRED_FILES = [
    "schwabot/init/omni_shell/lotus_omni_mesh.py",
    "schwabot/init/omni_shell/mesh_to_shell_sync.py",
    "schwabot/init/omni_shell/lotus_tick_hash_feed.py"
]

def check_modules() -> None:
    print("Checking required Python modules...")
    for mod in REQUIRED_MODULES:
        try:
            importlib.import_module(mod)
            print(f"  [OK] {mod}")
        except ImportError:
            print(f"  [MISSING] {mod}")

def check_files() -> None:
    print("Checking required mesh/omni files...")
    for f in REQUIRED_FILES:
        if os.path.isfile(f):
            print(f"  [OK] {f}")
        else:
            print(f"  [MISSING] {f}")

if __name__ == "__main__":
    check_modules()
    check_files()
    print("Mesh/Omni port check complete.") 