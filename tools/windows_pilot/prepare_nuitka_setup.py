"""Prepare a freshly compiled AutoPyro.dist for AutoPyro-Nuitka.iss."""

import argparse
import hashlib
import json
import shutil
import sys
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(bundle, pilot, release):
    candidate = json.loads((pilot / "candidate.json").read_text(encoding="utf-8"))
    if sha256(release) != candidate["archive_sha256"]:
        raise ValueError("Candidate wheel archive SHA-256 mismatch")
    if not (bundle / "AutoPyro.exe").is_file() or (bundle / "checks").exists():
        raise ValueError("Use a fresh standalone build")
    control = bundle / "control"
    control.mkdir(exist_ok=True)
    for name in ("candidate.json", "reference.json", "control.apyro"):
        shutil.copy2(pilot / name, control / name)
    with ZipFile(release) as archive:
        for name in archive.namelist():
            if name.startswith("examples/") or name == "dependency-licenses.csv":
                archive.extract(name, bundle)
            if name.endswith(".whl"):
                package = Path(name).name.split("-")[0]
                with ZipFile(BytesIO(archive.read(name))) as wheel:
                    for member in wheel.namelist():
                        path = Path(member)
                        if member.endswith("/"):
                            continue
                        if ("licenses" in [part.lower() for part in path.parts]
                                or path.name.lower().startswith(("license", "copying", "notice", "copyright"))):
                            wheel.extract(member, bundle / "licenses" / package)
    shutil.copy2(Path(sys.base_prefix) / "LICENSE.txt", bundle / "licenses" / "Python-LICENSE.txt")
    (bundle / "check-installed.cmd").write_bytes(
        b'@echo off\r\nsetlocal\r\n"%~dp0AutoPyro.exe" --check-installation\r\n'
        b'set "result=%errorlevel%"\r\nif "%result%"=="0" (echo PASS) else (echo FAILED: %result%)\r\n'
        b'echo "%~dp0checks"\r\nif /i not "%~1"=="--no-pause" pause\r\nexit /b %result%\r\n'
    )
    manifest = control / "installed-files.json"
    files = {path.relative_to(bundle).as_posix(): sha256(path) for path in sorted(bundle.rglob("*"))
             if path.is_file() and path != manifest}
    manifest.write_text(json.dumps(files, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("release", type=Path)
    args = parser.parse_args()
    prepare(args.bundle, args.pilot, args.release)
