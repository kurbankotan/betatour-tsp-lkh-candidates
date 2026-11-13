"""Build the v11 instrumented first-trial hook from pristine LKH-3.0.14.

Only our two headers and exact-source patches are shipped. Upstream LKH is
downloaded separately by the notebook and verified before modification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path


UPSTREAM_SHA256 = {
    "FindTour.c": "567e72acc9b7392488dd158be4df0267560424204a0e88d2d9b62f923a7c0eee",
    "Best2OptMove.c": "175ade008aa825db896a3fd121e14691bec983a4b539cb69a9d6082ffa117ead",
    "Best3OptMove.c": "35abfe9c2e66fb08e3c15298556cf53801058c53e6551b05c05b48481fc0335f",
    "Best4OptMove.c": "708ac57566fbbd2358e23036b9870e5c77f3efd600b2f21b80eed3cb98111bf9",
    "Best5OptMove.c": "09a7b85d200a3a61f1106b43531ea6636c5aad397bf9ee864bb3692fce670bcb",
    "BestKOptMove.c": "6088dbd3e7a36f42a24df17edd6ad2ee77b4b2207bef5455b9b8fcdc38b32c76",
    "BestSpecialOptMove.c": "efbdfb5ae9a12b816e3734901969e59f31bf019d0713b23c8bd19e4aaf06d301",
    "Gain23.c": "853d881b99c6035e0c08edb58fd5d4e4437f88c15415d8d182db19427ed9d3ab",
    "BridgeGain.c": "d625e4d63c192037fc540f7249d3ed62a5f8f7e2e44f108d5ab70c9b314bb963",
    "PatchCycles.c": "253e47b1bb64c11d76e372412b8339411e4d0792411be937e46667895bf60abc",
    "StoreTour.c": "683255360b8d9d71f3904898052ca47ae1bef5fea77c2311a20d169671166f74",
}

MOVE_LOOP_COUNTS = {
    "Best2OptMove.c": 1, "Best3OptMove.c": 2,
    "Best4OptMove.c": 3, "Best5OptMove.c": 4,
    "BestKOptMove.c": 1, "BestSpecialOptMove.c": 2,
    "Gain23.c": 3, "BridgeGain.c": 3, "PatchCycles.c": 3,
}
TOP_LEVEL_FILES = {
    "Best2OptMove.c", "Best3OptMove.c", "Best4OptMove.c",
    "Best5OptMove.c", "BestKOptMove.c", "BestSpecialOptMove.c",
}

_CANDIDATE_LOOP = re.compile(
    r"(?m)^([ \t]*)for\s*\(\s*(N\w+)\s*=\s*(\w+)->CandidateSet\s*;"
    r"\s*\(\s*(\w+)\s*=\s*\2->To\s*\)\s*;\s*\2\+\+\s*\)\s*\{"
)


def _replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"Expected one source anchor, found {count}: {old[:70]!r}")
    return source.replace(old, new, 1)


def patch_findtour(source: str) -> str:
    source = _replace_once(source, '#include "LKH.h"\n',
                           '#include "LKH.h"\n#include "BetaTourSearchHook.h"\n')
    source = _replace_once(source,
                           "    double EntryTime = GetTime();\n\n    t = FirstNode;",
                           "    double EntryTime = GetTime();\n\n"
                           "    BT_ExportNativeCandidates();\n    t = FirstNode;")
    source = _replace_once(source, "            RecordBetterTour();\n",
                           "            RecordBetterTour();\n            BT_RecordBest();\n")
    source = _replace_once(source,
                           "        }\n    }\n    if (BackboneTrials > 0 && BackboneTrials < MaxTrials) {",
                           "        }\n        BT_AfterFirstTrial();\n"
                           "    }\n    if (BackboneTrials > 0 && BackboneTrials < MaxTrials) {")
    return source


def patch_move_file(source: str, filename: str) -> str:
    source = _replace_once(source, '#include "LKH.h"\n',
                           '#include "LKH.h"\n#include "BetaTourVisit.h"\n')

    def insert_visit(match: re.Match[str]) -> str:
        indent, _candidate, from_name, to_name = match.groups()
        return match.group(0) + f"\n{indent}    BT_NoteCandidate({from_name}, {to_name});"

    source, count = _CANDIDATE_LOOP.subn(insert_visit, source)
    if count != MOVE_LOOP_COUNTS[filename]:
        raise ValueError(f"Unexpected candidate loops in {filename}: {count}")
    if filename in TOP_LEVEL_FILES:
        old = "BT_NoteCandidate(t2, t3);"
        source = _replace_once(source, old,
                               old + "\n        BT_NoteTopLevelEnumerated(t2, t3);")
        first = source.index("BT_NoteTopLevelEnumerated(t2, t3);")
        gate = "            continue;\n"
        end = source.find(gate, first)
        if end < 0:
            raise ValueError(f"Cannot find first-stage gain gate in {filename}")
        end += len(gate)
        source = source[:end] + "        BT_NoteTopLevelPass(t2, t3, G1);\n" + source[end:]
    return source


def patch_storetour(source: str) -> str:
    source = _replace_once(source, '#include "LKH.h"\n',
                           '#include "LKH.h"\n#include "BetaTourVisit.h"\n')
    source = _replace_once(source,
                           "            Activate(t);\n            t->OldPred = t->Pred;",
                           "            Activate(t);\n"
                           "            BT_NoteAcceptedEdge(t);\n"
                           "            t->OldPred = t->Pred;")
    return source


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def inspect_candidate_reordering(src: Path) -> dict:
    """Record, without patching, how upstream LKH reorders candidate rows.

    FindTour calls AdjustCandidateSet after each better tour, so an action's
    initial slot is not stable. The v11 hook enforces the slot as a floor at
    trial boundaries; this static check documents which upstream reordering
    rule the slot floor counteracts. It never fails the build.
    """
    report: dict = {}
    texts = {}
    for name in ("FindTour.c", "AdjustCandidateSet.c", "ResetCandidateSet.c"):
        path = src / name
        if path.exists():
            data = path.read_bytes()
            texts[name] = _compact(data.decode("utf-8", errors="replace"))
            report[name.replace(".c", "_sha256")] = hashlib.sha256(data).hexdigest()
        else:
            report[name.replace(".c", "_sha256")] = None
    findtour = texts.get("FindTour.c", "")
    loop = findtour.split("for(Trial=1;Trial<=MaxTrials;Trial++)", 1)
    report["findtour_trial_loop_calls_adjust_candidate_set"] = (
        len(loop) == 2 and "AdjustCandidateSet();" in loop[1].split("ResetCandidateSet();", 1)[0])
    adjust = texts.get("AdjustCandidateSet.c", "")
    report["adjust_moves_edges_of_two_best_tours_to_front"] = (
        "InBestTour(From,To)&&InNextBestTour(From,To)" in adjust)
    report["adjust_sorts_rows_by_alpha_then_cost"] = (
        "Temp.Alpha<NN->Alpha" in adjust)
    report["reset_sorts_rows_by_alpha_then_cost"] = (
        "Temp.Alpha<NN->Alpha" in texts.get("ResetCandidateSet.c", ""))
    return report


def build(source_dir: Path, output_dir: Path, jobs: int = 4) -> Path:
    source_dir, output_dir = source_dir.resolve(), output_dir.resolve()
    src = source_dir / "SRC"
    if not (source_dir / "Makefile").exists():
        raise FileNotFoundError(f"LKH source tree missing: {source_dir}")
    originals = {}
    for filename, expected in UPSTREAM_SHA256.items():
        data = (src / filename).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected:
            raise ValueError(f"Unexpected upstream SHA256 for {filename}: {digest}")
        originals[filename] = data.decode("utf-8")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    shutil.copytree(source_dir, output_dir,
                    ignore=shutil.ignore_patterns("OBJ", "LKH", "*.o"))
    (output_dir / "SRC" / "OBJ").mkdir(exist_ok=True)
    (output_dir / "SRC" / "FindTour.c").write_text(
        patch_findtour(originals["FindTour.c"]))
    for filename in MOVE_LOOP_COUNTS:
        (output_dir / "SRC" / filename).write_text(
            patch_move_file(originals[filename], filename))
    (output_dir / "SRC" / "StoreTour.c").write_text(
        patch_storetour(originals["StoreTour.c"]))
    for header in ("BetaTourSearchHook.h", "BetaTourVisit.h"):
        shutil.copy2(Path(__file__).with_name(header), output_dir / "SRC" / header)
    result = subprocess.run(["make", "-j", str(jobs)], cwd=output_dir,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"LKH instrumented build failed:\n{result.stdout[-8000:]}")
    binary = output_dir / "LKH"
    if not binary.exists():
        raise FileNotFoundError(f"Missing built binary: {binary}")
    manifest = {"upstream_sha256": UPSTREAM_SHA256,
                "search_hook_sha256": hashlib.sha256(
                    Path(__file__).with_name("BetaTourSearchHook.h").read_bytes()).hexdigest(),
                "visit_header_sha256": hashlib.sha256(
                    Path(__file__).with_name("BetaTourVisit.h").read_bytes()).hexdigest(),
                "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "instrumented_move_loops": sum(MOVE_LOOP_COUNTS.values()),
                "action_slot_rule": "floor enforced at every completed-trial boundary",
                "upstream_candidate_reordering": inspect_candidate_reordering(src)}
    (output_dir / "betatour_build.json").write_text(json.dumps(manifest, indent=2) + "\n")
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
