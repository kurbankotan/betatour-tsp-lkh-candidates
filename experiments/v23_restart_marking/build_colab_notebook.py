"""Build the self-contained, resumable run-v23 notebook (exploratory run on restart markings).

    python experiments/v23_restart_marking/build_colab_notebook.py

writes notebooks/25_BetaTour_v23_restart_marking.ipynb from the cell sources in cells/. Two modules are embedded
unchanged in %%writefile cells: the module of run v21 (../v21_alpha_encoding/alphaenc.py) and the marking module of
this run (marking.py). The configuration cell holds their SHA-256 values, the set-up cell compares the written files
with them, and the files are copied to the result archive. The configuration cell also holds one SHA-256 over all
code cells (NOTEBOOK_BUILD); it is stored in the manifest, and a later session with different cells is refused.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUTPUT = REPO / "notebooks" / "25_BetaTour_v23_restart_marking.ipynb"
# (file written in the notebook, source in the repository, placeholder in the configuration cell)
MODULES = [("alphaenc_v21.py", HERE.parent / "v21_alpha_encoding" / "alphaenc.py", "__CODE_SHA256__"),
           ("marking_v23.py", HERE / "marking.py", "__MARKING_SHA256__")]
CODE_CELLS = ["c1_config.py", 0, 1, "c3_setup.py", "c4_instances.py", "c5_experiment.py", "c6_selfcheck.py",
              "c7_run.py", "c8_analysis.py", "c9_archive.py"]          # integers: the embedded modules


def module_source(path: Path) -> str:
    text = path.read_text(encoding="ascii")
    assert text.endswith("\n") and not text.endswith("\n\n") and "\r" not in text and "\t" not in text
    compile(text, path.name, "exec")
    return text


def cell(kind: str, index: int, source: str) -> dict:
    out = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True),
           "id": f"v23_{index:02d}_{hashlib.sha256(source.encode()).hexdigest()[:10]}"}
    if kind == "code":
        out.update(execution_count=None, outputs=[])
    return out


def make_notebook() -> tuple:
    modules = [module_source(path) for _, path, _ in MODULES]
    digests = [hashlib.sha256(text.encode("ascii")).hexdigest() for text in modules]
    sources = {name: (HERE / "cells" / name).read_text(encoding="utf-8") for name in CODE_CELLS if isinstance(name, str)}
    build = hashlib.sha256(("\n".join(sources[name] for name in sorted(sources)) + "".join(modules))
                           .encode("utf-8")).hexdigest()
    cells = [cell("markdown", 0, (HERE / "cells" / "c0_intro.md").read_text(encoding="utf-8").rstrip("\n"))]
    for index, name in enumerate(CODE_CELLS, 1):
        if isinstance(name, int):
            # IPython writes the cell body, which ends with the module's own final newline, byte for byte.
            source = f"%%writefile {MODULES[name][0]}\n" + modules[name]
        else:
            source = sources[name]
            if name == "c1_config.py":
                for (_, _, placeholder), digest in zip(MODULES, digests):
                    assert source.count(placeholder) == 1
                    source = source.replace(placeholder, digest)
                assert source.count("__NOTEBOOK_BUILD__") == 1
                source = source.replace("__NOTEBOOK_BUILD__", build)
            compile(source, name, "exec")
            source = source.rstrip("\n")
        cells.append(cell("code", index, source))
    notebook = {"cells": cells,
                "metadata": {"accelerator": "GPU",
                             "colab": {"gpuType": "A100", "machine_shape": "hm", "provenance": []},
                             "kernelspec": {"display_name": "Python 3", "name": "python3"},
                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    return notebook, digests, build


if __name__ == "__main__":
    notebook, digests, build = make_notebook()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("written", OUTPUT)
    for (file_name, _, _), digest in zip(MODULES, digests):
        print(" ", file_name, "SHA-256", digest)
    print("  NOTEBOOK_BUILD", build)
