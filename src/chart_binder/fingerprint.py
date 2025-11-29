"""
Audio fingerprint utilities for acoustic identity.

Provides fingerprint generation via fpcalc (Chromaprint) and
fingerprint-based file identity for stable file tracking.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FingerprintResult:
    """Result of fingerprint calculation."""

    fingerprint: str
    duration_sec: int

    def generate_file_id(self) -> str:
        """
        Generate stable file ID from fingerprint and duration.

        This creates an acoustic identity that survives:
        - Tag edits (metadata changes don't affect audio)
        - File renames (path is not part of identity)
        - File moves (same audio = same identity)
        """
        signature = f"{self.fingerprint}:{self.duration_sec}"
        return hashlib.sha256(signature.encode()).hexdigest()


class FingerprintError(Exception):
    """Error during fingerprint calculation."""

    pass


def get_fpcalc_path() -> Path | None:
    """Find fpcalc executable in PATH."""
    path = shutil.which("fpcalc")
    return Path(path) if path else None


def calculate_fingerprint(
    file_path: Path,
    fpcalc_path: Path | None = None,
    timeout_sec: int = 30,
) -> FingerprintResult:
    """
    Calculate Chromaprint fingerprint for audio file.

    Args:
        file_path: Path to audio file
        fpcalc_path: Optional path to fpcalc executable
        timeout_sec: Timeout for fpcalc execution

    Returns:
        FingerprintResult with fingerprint and duration

    Raises:
        FingerprintError: If fpcalc is not available or fails
    """
    if fpcalc_path is None:
        fpcalc_path = get_fpcalc_path()

    if fpcalc_path is None:
        raise FingerprintError(
            "fpcalc not found in PATH. Install chromaprint-tools or "
            "download from https://acoustid.org/chromaprint"
        )

    try:
        result = subprocess.run(
            [str(fpcalc_path), "-json", str(file_path)],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )

        if result.returncode != 0:
            raise FingerprintError(f"fpcalc failed: {result.stderr.strip()}")

        data = json.loads(result.stdout)
        fingerprint = data.get("fingerprint")
        duration = data.get("duration")

        if not fingerprint or duration is None:
            raise FingerprintError(f"Invalid fpcalc output: {result.stdout}")

        return FingerprintResult(
            fingerprint=fingerprint,
            duration_sec=int(duration),
        )

    except subprocess.TimeoutExpired as e:
        raise FingerprintError(f"fpcalc timed out after {timeout_sec}s") from e
    except json.JSONDecodeError as e:
        raise FingerprintError(f"Failed to parse fpcalc output: {e}") from e
    except Exception as e:
        raise FingerprintError(f"Fingerprint calculation failed: {e}") from e


def generate_file_id_from_fingerprint(fingerprint: str, duration_sec: int) -> str:
    """
    Generate stable file ID from fingerprint and duration.

    This is the primary identity mechanism for files in the decision store.
    """
    result = FingerprintResult(fingerprint=fingerprint, duration_sec=duration_sec)
    return result.generate_file_id()


def generate_file_id_fallback(path: Path, size: int, mtime: float) -> str:
    """
    Generate fallback file ID from path, size, and mtime.

    Used when fingerprinting is not available or fails.
    Note: This ID is fragile - changes when tags are modified.
    """
    signature = f"fallback:{path}:{size}:{int(mtime)}"
    return hashlib.sha256(signature.encode()).hexdigest()


## Tests


def test_fingerprint_result_generate_file_id():
    """Test file ID generation is deterministic."""
    fp1 = FingerprintResult(fingerprint="ABC123", duration_sec=180)
    fp2 = FingerprintResult(fingerprint="ABC123", duration_sec=180)

    assert fp1.generate_file_id() == fp2.generate_file_id()

    # Different fingerprint = different ID
    fp3 = FingerprintResult(fingerprint="XYZ789", duration_sec=180)
    assert fp1.generate_file_id() != fp3.generate_file_id()

    # Different duration = different ID
    fp4 = FingerprintResult(fingerprint="ABC123", duration_sec=200)
    assert fp1.generate_file_id() != fp4.generate_file_id()


def test_generate_file_id_from_fingerprint():
    """Test helper function for file ID generation."""
    fp_result = FingerprintResult(fingerprint="ABC123", duration_sec=180)
    direct_id = fp_result.generate_file_id()

    helper_id = generate_file_id_from_fingerprint("ABC123", 180)
    assert direct_id == helper_id


def test_generate_file_id_fallback():
    """Test fallback file ID is prefixed and deterministic."""
    path = Path("/music/song.mp3")
    id1 = generate_file_id_fallback(path, 1024, 1234567890.0)
    id2 = generate_file_id_fallback(path, 1024, 1234567890.0)
    assert id1 == id2

    # Different params = different ID
    id3 = generate_file_id_fallback(path, 1025, 1234567890.0)
    assert id1 != id3
