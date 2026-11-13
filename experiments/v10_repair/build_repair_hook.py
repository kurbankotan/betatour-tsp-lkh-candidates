"""Build the delayed-barrier BetaTour v10 hook from LKH-3.0.14.

The LKH source is fetched separately by the experiment notebook. This script
ships only the BetaTour patch and checks the upstream file before editing it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


UPSTREAM_FINDTOUR_SHA256 = (
    "567e72acc9b7392488dd158be4df0267560424204a0e88d2d9b62f923a7c0eee"
)


def _replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"Expected one FindTour anchor, found {count}: {old[:80]!r}")
    return source.replace(old, new, 1)


def patch_findtour(source: str) -> str:
    source = _replace_once(
        source,
        '#include "LKH.h"\n',
        '#include "LKH.h"\n#include "BetaTourHook.h"\n',
    )
    source = _replace_once(
        source,
        "    double EntryTime = GetTime();\n\n    t = FirstNode;",
        "    double EntryTime = GetTime();\n\n"
        "    BT_ExportNativeCandidates();\n    t = FirstNode;",
    )
    source = _replace_once(
        source,
        "            RecordBetterTour();\n",
        "            RecordBetterTour();\n            BT_RecordBest();\n",
    )
    source = _replace_once(
        source,
        "        }\n    }\n    if (BackboneTrials > 0 && BackboneTrials < MaxTrials) {",
        "        }\n        BT_AfterFirstTrial();\n"
        "    }\n    if (BackboneTrials > 0 && BackboneTrials < MaxTrials) {",
    )
    return source


def build(source_dir: Path, output_dir: Path, jobs: int = 4) -> Path:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    upstream = source_dir / "SRC" / "FindTour.c"
    if not upstream.exists() or not (source_dir / "Makefile").exists():
        raise FileNotFoundError(f"LKH-3.0.14 source tree missing: {source_dir}")
    original_bytes = upstream.read_bytes()
    digest = hashlib.sha256(original_bytes).hexdigest()
    if digest != UPSTREAM_FINDTOUR_SHA256:
        raise ValueError(f"Unexpected LKH FindTour.c SHA256: {digest}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    shutil.copytree(
        source_dir,
        output_dir,
        ignore=shutil.ignore_patterns("OBJ", "LKH", "*.o"),
    )
    (output_dir / "SRC" / "OBJ").mkdir(exist_ok=True)
    patched = patch_findtour(original_bytes.decode("utf-8"))
    (output_dir / "SRC" / "FindTour.c").write_text(patched)
    hook = Path(__file__).with_name("BetaTourRepairHook.h")
    shutil.copy2(hook, output_dir / "SRC" / "BetaTourHook.h")
    result = subprocess.run(
        ["make", "-j", str(jobs)],
        cwd=output_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"LKH hook build failed:\n{result.stdout[-6000:]}")
    binary = output_dir / "LKH"
    if not binary.exists():
        raise FileNotFoundError(f"Patched binary missing: {binary}")
    manifest = {
        "upstream_findtour_sha256": digest,
        "patch_header_sha256": hashlib.sha256(hook.read_bytes()).hexdigest(),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
    }
    (output_dir / "betatour_build.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return binary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    print(build(args.source_dir, args.output_dir, args.jobs))


if __name__ == "__main__":
    main()
