"""Dependency-free export integrity and syntax checks; not a GPU smoke test."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text(encoding="utf-8-sig"))
errors = []
for entry in manifest["kept_files"]:
    path = root / entry["path"]
    if not path.is_file():
        errors.append(f"Missing: {entry['path']}")
    elif hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        errors.append(f"Changed since export: {entry['path']}")
for directory in ("verl", "agent_system", "gigpo"):
    for path in (root / directory).rglob("*.py"):
        try:
            ast.parse(path.read_bytes(), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f"Syntax: {path.relative_to(root)}: {exc}")
for directory in ("recipe", "docker", ".github"):
    if (root / directory).exists():
        errors.append(f"Unexpected excluded directory: {directory}")
if errors:
    raise SystemExit("\n".join(errors))
print(f"PASS: {len(manifest['kept_files'])} source hashes and Python syntax")
print("ARPO/TAPO are not implemented. GPU training has not been tested by this check.")
