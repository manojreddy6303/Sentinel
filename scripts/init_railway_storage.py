"""
Sentinel Storage & Database Initializer for Railway Persistent Volume.
Ensures /storage directories exist and seeds /storage/sentinel.db on initial volume mount without overwriting existing data.
"""
import os
import shutil
from pathlib import Path

def init_railway_storage():
    storage_base_env = os.getenv("STORAGE_BASE_DIR", "./storage")
    storage_base = Path(storage_base_env).resolve()
    
    # 1. Ensure required subdirectories exist
    subdirs = [
        "uploads",
        "evidence",
        "evidence_playback",
        "playback",
        "reports",
        "events",
    ]
    for sub in subdirs:
        (storage_base / sub).mkdir(parents=True, exist_ok=True)
    
    # 2. Database Initialization
    target_db = storage_base / "sentinel.db"
    repo_root = Path(__file__).resolve().parent.parent
    seed_db = repo_root / "storage" / "sentinel.db"
    
    if not target_db.exists():
        if seed_db.exists() and seed_db.resolve() != target_db:
            print(f"[STORAGE_INIT] Initializing persistent database at {target_db} from seed...")
            shutil.copy2(str(seed_db), str(target_db))
            print("[STORAGE_INIT] Persistent database initialized successfully.")
        else:
            print("[STORAGE_INIT] Target database already at seed location or seed unavailable.")
    else:
        print(f"[STORAGE_INIT] Persistent database exists at {target_db}. Preserving existing volume data.")

    # 3. Seed canonical uploads & evidence files if destination is an external mount (/storage)
    if seed_db.resolve() != target_db:
        for sub in ["uploads", "evidence", "evidence_playback"]:
            src_dir = repo_root / "storage" / sub
            dst_dir = storage_base / sub
            if src_dir.exists():
                for f in src_dir.glob("*"):
                    dst_file = dst_dir / f.name
                    if not dst_file.exists() and f.is_file():
                        try:
                            shutil.copy2(str(f), str(dst_file))
                        except Exception:
                            pass

if __name__ == "__main__":
    init_railway_storage()
