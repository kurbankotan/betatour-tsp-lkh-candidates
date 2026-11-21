"""Build the self-contained, resumable run-v27 notebook (pilot: exact merging on the union graph of the tours).

    python experiments/v27_union_exact/build_colab_notebook.py

writes notebooks/29_BetaTour_v27_union_exact_pilot.ipynb from the cell sources in cells/. Two modules are embedded
unchanged in %%writefile cells: the module of run v26 (../v26_recombination_confirmation/byprod.py, written as
byprod_v26.py) and unionx.py (written as unionx_v27.py). The configuration cell holds their SHA-256 values, the
SHA-256 of experiments/PROTOCOL_v27.md and one SHA-256 over all code cells, both modules and the protocol
(NOTEBOOK_BUILD); the set-up cell compares the written files with them. Before the first instance the set-up cell
requires that the protocol on the main branch on GitHub has that SHA-256 and that the notebook on GitHub carries that
build hash, so the protocol and the notebook must be committed and pushed before the run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUTPUT = REPO / "notebooks" / "29_BetaTour_v27_union_exact_pilot.ipynb"
PROTOCOL = REPO / "experiments" / "PROTOCOL_v27.md"
MODULES = [("byprod_v26.py", HERE.parent / "v26_recombination_confirmation" / "byprod.py", "__CODE_SHA256__"),
           ("unionx_v27.py", HERE / "unionx.py", "__UNION_SHA256__")]
CODE_CELLS = ["c1_config.py", 0, 1, "c3_setup.py", "c4_instances.py", "c5_experiment.py", "c6_selfcheck.py",
              "c7_run.py", "c8_analysis.py", "c9_archive.py"]          # integers: the embedded modules


def module_source(path: Path) -> str:
    text = path.read_text(encoding="ascii")
    assert text.endswith("\n") and not text.endswith("\n\n") and "\r" not in text and "\t" not in text, path
    compile(text, path.name, "exec")
    return text


def protocol_sha256() -> str:
    data = PROTOCOL.read_bytes()
    assert b"\r" not in data and data.endswith(b"\n"), "the protocol must have LF line endings and end with one"
    return hashlib.sha256(data).hexdigest()


def cell(kind: str, index: int, source: str) -> dict:
    out = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True),
           "id": f"v27_{index:02d}_{hashlib.sha256(source.encode()).hexdigest()[:10]}"}
    if kind == "code":
        out.update(execution_count=None, outputs=[])
    return out


def make_notebook() -> dict:
    modules = [module_source(path) for _, path, _ in MODULES]
    digests = [hashlib.sha256(text.encode("ascii")).hexdigest() for text in modules]
    protocol = protocol_sha256()
    sources = {name: (HERE / "cells" / name).read_text(encoding="utf-8") for name in CODE_CELLS if isinstance(name, str)}
    build = hashlib.sha256(("\n".join(sources[name] for name in sorted(sources)) + "".join(modules) + protocol)
                           .encode("utf-8")).hexdigest()
    cells = [cell("markdown", 0, (HERE / "cells" / "c0_intro.md").read_text(encoding="utf-8").rstrip("\n"))]
    for index, name in enumerate(CODE_CELLS, 1):
        if isinstance(name, int):
            # IPython writes the cell body, which ends with the module's own final newline, byte for byte.
            source = f"%%writefile {MODULES[name][0]}\n" + modules[name]
        else:
            source = sources[name]
            if name == "c1_config.py":
                markers = [placeholder for _, _, placeholder in MODULES] + ["__NOTEBOOK_BUILD__", "__PROTOCOL_SHA256__"]
                for marker in markers:
                    assert source.count(marker) == 1, marker
                for (_, _, placeholder), digest in zip(MODULES, digests):
                    source = source.replace(placeholder, digest)
                source = source.replace("__NOTEBOOK_BUILD__", build).replace("__PROTOCOL_SHA256__", protocol)
            compile(source, name, "exec")
            source = source.rstrip("\n")
        cells.append(cell("code", index, source))
    return {"cells": cells,
            "metadata": {"accelerator": "GPU",
                         "colab": {"gpuType": "L4", "machine_shape": "hm", "provenance": []},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(make_notebook(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for (file_name, path, _), text in zip(MODULES, [module_source(p) for _, p, _ in MODULES]):
        print(file_name, "SHA-256", hashlib.sha256(text.encode("ascii")).hexdigest())
    print("written", OUTPUT, "| protocol SHA-256", protocol_sha256())
