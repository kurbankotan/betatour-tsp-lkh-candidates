"""Build the self-contained, resumable run-v26 notebook (confirmation run: recombining the tours of LKH-3 processes).

    python experiments/v26_recombination_confirmation/build_colab_notebook.py

writes notebooks/28_BetaTour_v26_recombination_confirmation.ipynb from the cell sources in cells/. The module
byprod.py is embedded unchanged in a %%writefile cell; the configuration cell holds its SHA-256, the set-up cell
compares the written file with it, and the file is copied to the result archive. The configuration cell also holds
the SHA-256 of experiments/PROTOCOL_v26.md and one SHA-256 over all code cells, the module and the protocol
(NOTEBOOK_BUILD). Before the first instance the set-up cell requires that the protocol on the main branch on GitHub
has that SHA-256 and that the notebook on GitHub carries that build hash, so the protocol and the notebook must be
committed and pushed before the run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUTPUT = REPO / "notebooks" / "28_BetaTour_v26_recombination_confirmation.ipynb"
PROTOCOL = REPO / "experiments" / "PROTOCOL_v26.md"
MODULE_NAME = "byprod_v26.py"
CODE_CELLS = ["c1_config.py", None, "c3_setup.py", "c4_instances.py", "c5_experiment.py", "c6_selfcheck.py",
              "c7_run.py", "c8_analysis.py", "c9_archive.py"]          # None: the embedded module


def module_source() -> str:
    text = (HERE / "byprod.py").read_text(encoding="ascii")
    assert text.endswith("\n") and not text.endswith("\n\n") and "\r" not in text and "\t" not in text
    compile(text, "byprod.py", "exec")
    return text


def protocol_sha256() -> str:
    data = PROTOCOL.read_bytes()
    assert b"\r" not in data and data.endswith(b"\n"), "the protocol must have LF line endings and end with one"
    return hashlib.sha256(data).hexdigest()


def cell(kind: str, index: int, source: str) -> dict:
    out = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True),
           "id": f"v26_{index:02d}_{hashlib.sha256(source.encode()).hexdigest()[:10]}"}
    if kind == "code":
        out.update(execution_count=None, outputs=[])
    return out


def make_notebook() -> dict:
    module = module_source()
    digest = hashlib.sha256(module.encode("ascii")).hexdigest()
    protocol = protocol_sha256()
    sources = {name: (HERE / "cells" / name).read_text(encoding="utf-8") for name in CODE_CELLS if name}
    build = hashlib.sha256(("\n".join(sources[name] for name in sorted(sources)) + module + protocol)
                           .encode("utf-8")).hexdigest()
    cells = [cell("markdown", 0, (HERE / "cells" / "c0_intro.md").read_text(encoding="utf-8").rstrip("\n"))]
    for index, name in enumerate(CODE_CELLS, 1):
        if name is None:
            # IPython writes the cell body, which ends with the module's own final newline, byte for byte.
            source = f"%%writefile {MODULE_NAME}\n" + module
        else:
            source = sources[name]
            if name == "c1_config.py":
                for marker in ("__CODE_SHA256__", "__NOTEBOOK_BUILD__", "__PROTOCOL_SHA256__"):
                    assert source.count(marker) == 1, marker
                source = (source.replace("__CODE_SHA256__", digest).replace("__NOTEBOOK_BUILD__", build)
                          .replace("__PROTOCOL_SHA256__", protocol))
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
    print("written", OUTPUT, "| module SHA-256", hashlib.sha256(module_source().encode("ascii")).hexdigest(),
          "| protocol SHA-256", protocol_sha256())
