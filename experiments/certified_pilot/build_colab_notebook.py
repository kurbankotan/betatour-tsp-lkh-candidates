"""Build the self-contained, resumable notebook of the certified-candidate-graph pilot (not a run of the study).

    python experiments/certified_pilot/build_colab_notebook.py

writes notebooks/32_Certified_candidate_graph_pilot.ipynb from the cell sources in cells/. The module certpilot.py is
embedded unchanged in a %%writefile cell; the configuration cell holds its SHA-256 and one SHA-256 over all code cells
and the module (NOTEBOOK_BUILD), and the set-up cell compares the written file with them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUTPUT = REPO / "notebooks" / "32_Certified_candidate_graph_pilot.ipynb"
MODULE = HERE / "certpilot.py"
CODE_CELLS = ["c1_config.py", 0, "c3_setup.py", "c4_functions.py", "c5_selfcheck.py", "c6_run.py", "c7_summary.py"]


def module_source(path: Path) -> str:
    text = path.read_text(encoding="ascii")
    assert text.endswith("\n") and not text.endswith("\n\n") and "\r" not in text and "\t" not in text, path
    compile(text, path.name, "exec")
    return text


def cell(kind: str, index: int, source: str) -> dict:
    out = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True),
           "id": f"cp1_{index:02d}_{hashlib.sha256(source.encode()).hexdigest()[:10]}"}
    if kind == "code":
        out.update(execution_count=None, outputs=[])
    return out


def make_notebook() -> dict:
    module = module_source(MODULE)
    digest = hashlib.sha256(module.encode("ascii")).hexdigest()
    sources = {name: (HERE / "cells" / name).read_text(encoding="utf-8") for name in CODE_CELLS if isinstance(name, str)}
    build = hashlib.sha256(("\n".join(sources[name] for name in sorted(sources)) + module).encode("utf-8")).hexdigest()
    cells = [cell("markdown", 0, (HERE / "cells" / "c0_intro.md").read_text(encoding="utf-8").rstrip("\n"))]
    for index, name in enumerate(CODE_CELLS, 1):
        if isinstance(name, int):
            source = "%%writefile certpilot.py\n" + module
        else:
            source = sources[name]
            if name == "c1_config.py":
                for marker in ("__CODE_SHA256__", "__NOTEBOOK_BUILD__"):
                    assert source.count(marker) == 1, marker
                source = source.replace("__CODE_SHA256__", digest).replace("__NOTEBOOK_BUILD__", build)
            compile(source, name, "exec")
            source = source.rstrip("\n")
        cells.append(cell("code", index, source))
    return {"cells": cells,
            "metadata": {"accelerator": "GPU",
                         "colab": {"gpuType": "L4", "machine_shape": "hm", "provenance": []},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    notebook = make_notebook()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", OUTPUT)


if __name__ == "__main__":
    main()
