# launcher.py
import os, sys, subprocess, importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
SHARE = ROOT / "src" / "share.py"

# Valeur par défaut depuis share.py si dispo
default_folder = "./NWB_files"
if SHARE.exists():
    try:
        spec = importlib.util.spec_from_file_location("share", str(SHARE))
        share = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(share)
        default_folder = getattr(share, "Folder", default_folder)
    except Exception:
        pass

# Demande du chemin dossier
folder_in = input(f"Folder path [{default_folder}]: ").strip() or default_folder
folder = str(Path(folder_in).expanduser().resolve())

if not Path(folder).is_dir():
    print(f"❌ Folder not found: {folder}")
    sys.exit(1)

# Écrit la variable Folder dans src/share.py (écrase le contenu si besoin)
SHARE.write_text(f'Folder = r"{folder}"\n', encoding="utf-8")
print(f"✅ Folder set to: {folder}")

# Lance l'app (module 'src.app')
subprocess.run([sys.executable, "-m", "src.app"], check=True)
