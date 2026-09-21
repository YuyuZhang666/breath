import json
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from future_war_agent.seclog import decrypt, is_encrypted


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = "FutureWarAgent"
BUILDER = ROOT / "tools" / "build_submission.py"


class SubmissionPackageTests(unittest.TestCase):
    def test_archive_contains_only_runtime_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = self._build_archive(Path(temp_dir))
            with tarfile.open(archive_path, "r:gz") as archive:
                names = {
                    member.name.rstrip("/")
                    for member in archive.getmembers()
                }

        self.assertIn(f"{PACKAGE_ROOT}/main3.py", names)
        self.assertIn(f"{PACKAGE_ROOT}/pyproject.toml", names)
        self.assertIn(
            f"{PACKAGE_ROOT}/src/future_war_agent/__init__.py",
            names,
        )
        self.assertIn(
            f"{PACKAGE_ROOT}/src/future_war_agent/seclog.py",
            names,
        )
        self.assertIn(
            f"{PACKAGE_ROOT}/src/future_war_agent/logtool.py",
            names,
        )
        self.assertIn(
            f'{PACKAGE_ROOT}/src/future_war_agent/deadline.py',
            names,
        )
        self.assertTrue(
            all(
                name == PACKAGE_ROOT
                or name.startswith(f"{PACKAGE_ROOT}/")
                for name in names
            )
        )

        forbidden_parts = {
            ".git",
            ".idea",
            "__pycache__",
            "docs",
            "tests",
        }
        for name in names:
            path = Path(name)
            self.assertTrue(forbidden_parts.isdisjoint(path.parts), name)
            self.assertNotEqual(path.suffix, ".pyc", name)

    def test_extracted_launcher_serves_a_match_request(self) -> None:
        request_body = (
            ROOT / "tests" / "fixtures" / "interface_request.json"
        ).read_bytes()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            archive_path = self._build_archive(temp_path)
            extract_dir = temp_path / "extracted"
            extract_dir.mkdir()
            with tarfile.open(archive_path, "r:gz") as archive:
                archive.extractall(extract_dir, filter="data")

            package_dir = extract_dir / PACKAGE_ROOT
            port = self._available_port()
            process = subprocess.Popen(
                [sys.executable, "main3.py", str(port)],
                cwd=package_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            output_lines: list[str] = []
            output_thread = threading.Thread(
                target=self._collect_output,
                args=(process, output_lines),
                daemon=True,
            )
            output_thread.start()
            try:
                payload = self._post_when_ready(port, request_body, process)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                output_thread.join(timeout=5)
                if process.stdout is not None:
                    process.stdout.close()

        self.assertEqual(
            {"roleCommandMap", "prompt", "executeCmd"},
            set(payload),
        )
        self.assertIsInstance(payload["roleCommandMap"], dict)
        encrypted_lines = [
            line.rstrip("\r\n")
            for line in output_lines
            if is_encrypted(line.rstrip("\r\n"))
        ]
        self.assertTrue(encrypted_lines, output_lines)
        self.assertTrue(
            any(
                "agent server listening" in decrypt(line)
                for line in encrypted_lines
            ),
            output_lines,
        )

    def _build_archive(self, output_dir: Path) -> Path:
        result = subprocess.run(
            [sys.executable, str(BUILDER), str(output_dir)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(
            0,
            result.returncode,
            f"builder failed:\n{result.stdout}\n{result.stderr}",
        )
        archive_path = output_dir / f"{PACKAGE_ROOT}.tar.gz"
        self.assertTrue(archive_path.is_file())
        return archive_path

    @staticmethod
    def _available_port() -> int:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            return int(listener.getsockname()[1])

    def _post_when_ready(
        self,
        port: int,
        body: bytes,
        process: subprocess.Popen[str],
    ) -> dict[str, object]:
        request = Request(
            f"http://127.0.0.1:{port}/",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        deadline = time.monotonic() + 10
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                self.fail("packaged server exited early")
            try:
                with urlopen(request, timeout=1) as response:
                    self.assertEqual(200, response.status)
                    return json.loads(response.read().decode("utf-8"))
            except (ConnectionError, TimeoutError, URLError) as error:
                last_error = error
                time.sleep(0.05)
        self.fail(f"packaged server did not become ready: {last_error}")

    @staticmethod
    def _collect_output(
        process: subprocess.Popen[str],
        output_lines: list[str],
    ) -> None:
        if process.stdout is None:
            return
        output_lines.extend(process.stdout)


if __name__ == "__main__":
    unittest.main()
