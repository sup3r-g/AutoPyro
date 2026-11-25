"""Install the bundled Windows candidate and run its portable control case."""

import hashlib
import json
import os
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory, mkdtemp
from zipfile import BadZipFile, ZipFile


def client_archive(root):
    kit = root / "control"
    candidate = json.loads((kit / "candidate.json").read_text(encoding="utf-8"))
    archive = root / candidate["archive"]
    print("Проверка SHA-256 установочного архива…", flush=True)
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != candidate["archive_sha256"]:
            raise ValueError("Контрольная сумма архива отличается. Распакуйте исходный комплект заново.")

    return archive


def install_environment(venv, archive, *, log=None):
    run = venv.parent
    python = venv / "Scripts" / "python.exe"
    subprocess.run([sys.executable, "-I", "-m", "venv", venv], check=True, stdout=log, stderr=log)
    pip = [python, "-I", "-m", "pip", "--isolated", "--disable-pip-version-check"]
    with TemporaryDirectory(dir=run, prefix="wheels-") as temp:
        with ZipFile(archive) as zipped:
            zipped.extractall(temp)
        subprocess.run([*pip, "install", "--no-index", "--find-links", "wheels", "--require-hashes",
                        "-r", "install.lock"], cwd=temp, check=True, stdout=log, stderr=log)
    subprocess.run([*pip, "check"], cwd=run, check=True, stdout=log, stderr=log)
    return python


def check_install(python, archive, result, *, log=None):
    kit = archive.parent / "control"
    subprocess.run([python, "-I", "-X", "utf8", kit / "verify_windows_pilot.py", "reference",
                    "--zip", archive, "--output", result], cwd=result.parent, check=True, stdout=log, stderr=log,
                   env={**os.environ, "QT_QPA_PLATFORM": "windows"})
    print(f"Контроль пройден. Протокол: {result / 'result.json'}")
    print(f"Запуск приложения: {python.with_name('autopyro.exe')}")


def main():
    root = Path(__file__).resolve().parent
    archive = client_archive(root)
    if sys.argv[1:] in (["setup"], ["check"]):
        venv = root / "env"
        checks = root / "checks"
        checks.mkdir(exist_ok=True)
        folder = Path(mkdtemp(dir=checks, prefix=datetime.now().strftime("%Y%m%d-%H%M%S-")))
        # Keep pip output and verifier errors even when launched from the wizard.
        with (folder / "install-check.log").open("w", encoding="utf-8") as log:
            with redirect_stdout(log), redirect_stderr(log):
                if sys.argv[1] == "setup":
                    install_environment(venv, archive, log=log)
                check_install(venv / "Scripts" / "python.exe", archive, folder / "control", log=log)
        return
    runs = root / "runs"
    runs.mkdir(exist_ok=True)
    run = Path(mkdtemp(dir=runs, prefix=datetime.now().strftime("%Y%m%d-%H%M%S-")))
    print(f"Папка установки и результатов: {run}", flush=True)
    python = install_environment(run / "venv", archive)
    check_install(python, archive, run / "control")


if __name__ == "__main__":
    try:
        main()
    except (OSError, BadZipFile, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"Установка или проверка прервана: {error}", file=sys.stderr)
        sys.exit(1)
