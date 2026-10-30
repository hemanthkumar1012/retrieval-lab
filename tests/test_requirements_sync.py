"""Vercel installs pyproject.toml dependencies, Docker installs requirements.txt. Keep them equal."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# the Vercel runtime brings its own ASGI server
DOCKER_ONLY = {"uvicorn[standard]==0.32.0"}


def test_pyproject_and_requirements_match():
    deps = set(tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"])
    reqs = {
        line.strip()
        for line in (ROOT / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert deps == reqs - DOCKER_ONLY
