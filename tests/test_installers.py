"""Offline checks for the public Foldseek installation entry points."""
from __future__ import annotations

import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_linux_installer_has_valid_shell_and_machine_neutral_gpu_default():
    installer = REPOSITORY_ROOT / "scripts" / "install_foldseek_linux.sh"
    subprocess.run(["bash", "-n", str(installer)], check=True)
    help_result = subprocess.run(
        ["bash", str(installer), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    text = installer.read_text()
    assert "--gpu | --cpu" in help_result.stdout
    assert 'FOLDSEEK_CUDA_ARCHITECTURES:-native' in text
    assert "89-real" not in text
    assert "8dc75c74ad0eddab73cfd905963d13bf74dc012b" in text


def test_windows_installer_is_a_wsl_launcher_for_the_linux_installer():
    installer = REPOSITORY_ROOT / "scripts" / "install_foldseek_windows.ps1"
    text = installer.read_text()
    assert "wsl.exe" in text
    assert "install_foldseek_linux.sh" in text
    assert "nvidia-smi" in text
    assert "native Windows binary" not in text
