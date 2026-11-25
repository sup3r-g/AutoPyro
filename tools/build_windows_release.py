"""Build an offline AutoPyro desktop bundle for Windows x64 and CPython 3.12."""

import argparse
import csv
from email.parser import BytesParser
from hashlib import file_digest
import io
from pathlib import Path
import shutil
import subprocess
import sys
from sysconfig import get_platform
from tempfile import NamedTemporaryFile, TemporaryDirectory
from zipfile import ZIP_STORED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "requirements-win-py312.lock"


def pip(*args):
    subprocess.run(
        [sys.executable, "-m", "pip", "--isolated", "--disable-pip-version-check", *map(str, args)],
        check=True,
    )


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def wheel_metadata(wheel):
    with ZipFile(wheel) as archive:
        name = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        return BytesParser().parsebytes(archive.read(name))


def license_register(wheels):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(("Package", "Version", "License expression", "License metadata", "License classifiers", "Wheel"))
    for wheel in sorted(wheels):
        metadata = wheel_metadata(wheel)
        classifiers = [item for item in metadata.get_all("Classifier", ()) if item.startswith("License ::")]
        writer.writerow((metadata["Name"], metadata["Version"], metadata.get("License-Expression", ""),
                         str(metadata.get("License", "")).strip(),
                         " | ".join(classifiers), wheel.name))
    return output.getvalue().encode("utf-8-sig")


def build(output_dir, wheelhouse):
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="autopyro-build-", dir=output_dir) as directory:
        staging = Path(directory)
        source = staging / "source"
        shutil.copytree(ROOT / "autopyro", source / "autopyro",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copy2(ROOT / "pyproject.toml", source)
        pip("wheel", "--no-deps", "--no-build-isolation", "--no-index", source, "-w", staging)
        wheel, = staging.glob("autopyro-*.whl")
        version = wheel_metadata(wheel)["Version"]
        output = output_dir / f"AutoPyro-{version}-windows-x64-py312.zip"
        if output.exists():
            raise FileExistsError(output)

        dependency_dir = staging / "wheels"
        fetch = ("--no-index", "--find-links", wheelhouse) if wheelhouse else ("--index-url", "https://pypi.org/simple")
        pip("download", "--only-binary=:all:", "--require-hashes", *fetch, "-r", LOCK, "-d", dependency_dir)
        dependencies = list(dependency_dir.glob("*.whl"))
        install_lock = LOCK.read_text(encoding="utf-8") + f"autopyro=={version} --hash=sha256:{digest(wheel)}\n"
        # Inherit the destination's permissions instead of the private staging directory's.
        with NamedTemporaryFile(dir=output_dir, suffix=".zip", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            # Wheels are already compressed. Keep them intact with their license notices.
            with ZipFile(temporary, "w", compression=ZIP_STORED) as archive:
                for file in sorted([*dependencies, wheel]):
                    archive.write(file, f"wheels/{file.name}")
                archive.writestr("install.lock", install_lock)
                archive.writestr("dependency-licenses.csv", license_register(dependencies))
                archive.write(ROOT / "docs" / "WINDOWS_INSTALL.md", "README.md")
                archive.write(ROOT / "examples" / "pyrolysis.csv", "examples/pyrolysis.csv")
            temporary.rename(output)
        finally:
            temporary.unlink(missing_ok=True)
        print(output)
        print(f"SHA-256 {digest(output)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    parser.add_argument("--wheelhouse", type=Path, help="Use previously downloaded dependency wheels")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12) or get_platform() != "win-amd64":
        parser.error("Build this release with Windows x64 CPython 3.12")
    build(args.output.resolve(), args.wheelhouse.resolve() if args.wheelhouse else None)


if __name__ == "__main__":
    main()
