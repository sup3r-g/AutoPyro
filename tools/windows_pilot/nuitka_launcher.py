"""Windows standalone entry point including the installed release check."""

import sys
from multiprocessing import freeze_support


def check_installation():
    import json
    import os
    import tempfile
    import traceback
    from datetime import datetime
    from pathlib import Path

    from verify_windows_pilot import check_reference_case, require, sha256
    from autopyro import __version__

    root = Path(sys.executable).resolve().parent
    checks = root / "checks"
    checks.mkdir(exist_ok=True)
    folder = Path(tempfile.mkdtemp(dir=checks, prefix=datetime.now().strftime("%Y%m%d-%H%M%S-")))
    kit = root / "control"
    try:
        manifest = json.loads((kit / "installed-files.json").read_text(encoding="utf-8"))
        for name, expected in manifest.items():
            path = root / name
            require(path.is_file() and sha256(path) == expected, f"Installed file differs: {name}")
        candidate = json.loads((kit / "candidate.json").read_text(encoding="utf-8"))
        require(candidate["version"] == __version__, "Unexpected AutoPyro release")
        os.environ["QT_QPA_PLATFORM"] = "windows"
        check_reference_case(kit, candidate, folder / "control")
    except Exception:
        (folder / "check.log").write_text(traceback.format_exc(), encoding="utf-8")
        return 1
    (folder / "check.log").write_text("PASS\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    freeze_support()
    if len(sys.argv) == 3 and sys.argv[1] == "--write-recovery":
        from autopyro.database.recovery_writer import writer_main

        sys.exit(writer_main(sys.argv[2]))
    if sys.argv[1:] == ["--check-installation"]:
        sys.exit(check_installation())
    from autopyro.ui.application import main

    sys.exit(main())
