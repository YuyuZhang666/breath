#!/usr/bin/env python3
import argparse
import shutil
import tarfile
import tempfile
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "FutureWarAgent"
RUNTIME_SOURCE = ROOT / "future_war_agent"

LAUNCHER = dedent(
    """\
    #!/usr/bin/env python3
    import logging
    import os
    import sys
    from collections.abc import Callable, Sequence
    from pathlib import Path


    USAGE = "Usage: python main3.py <port>"


    def parse_port(argv: Sequence[str]) -> int:
        if len(argv) != 1:
            raise SystemExit(USAGE)
        try:
            port = int(argv[0])
        except ValueError as error:
            raise SystemExit(USAGE) from error
        if not 1 <= port <= 65535:
            raise SystemExit(USAGE)
        return port


    def main(
        argv: Sequence[str] | None = None,
        runner: Callable[[int], None] | None = None,
    ) -> int:
        root = Path(__file__).resolve().parent
        os.chdir(root)
        sys.path.insert(0, str(root / "src"))

        from future_war_agent.server import serve

        port = parse_port(sys.argv[1:] if argv is None else argv)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            stream=sys.stdout,
        )
        (serve if runner is None else runner)(port)
        return 0


    if __name__ == "__main__":
        raise SystemExit(main())
    """
)

PYPROJECT = dedent(
    """\
    [build-system]
    requires = ["setuptools>=68"]
    build-backend = "setuptools.build_meta"

    [project]
    name = "future-war-agent"
    version = "1.0.0"
    requires-python = ">=3.11"
    dependencies = []

    [tool.setuptools]
    package-dir = {"" = "src"}

    [tool.setuptools.packages.find]
    where = ["src"]
    """
)


def build_submission(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"{PACKAGE_NAME}.tar.gz"
    if archive_path.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {archive_path}")
    if not RUNTIME_SOURCE.is_dir():
        raise FileNotFoundError(f"runtime package not found: {RUNTIME_SOURCE}")

    with tempfile.TemporaryDirectory(prefix="future-war-submission-") as temp_dir:
        temp_path = Path(temp_dir)
        package_root = temp_path / PACKAGE_NAME
        runtime_target = package_root / "src" / RUNTIME_SOURCE.name
        shutil.copytree(
            RUNTIME_SOURCE,
            runtime_target,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        (package_root / "main3.py").write_text(LAUNCHER, encoding="utf-8")
        (package_root / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")

        temporary_archive = temp_path / archive_path.name
        with tarfile.open(temporary_archive, "w:gz") as archive:
            archive.add(package_root, arcname=PACKAGE_NAME)
        shutil.copy2(temporary_archive, archive_path)

    return archive_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the Future War competition submission archive.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory that will receive FutureWarAgent.tar.gz",
    )
    args = parser.parse_args()
    archive_path = build_submission(args.output_dir)
    print(archive_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
