"""Generate a self-contained Colab Pro notebook for the v15 live-swap pilot.

The frozen source and checkpoint are read from a pinned Git commit.  The new
experiment code is embedded in the notebook and copied verbatim to Drive.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUTPUT = REPO / "notebooks" / "15_BetaTour_v15_live_swap_pilot.ipynb"
V11_BUILDER = HERE.parent / "v11_candidate" / "build_colab_notebook.py"
spec = importlib.util.spec_from_file_location("v11_builder_for_v15", V11_BUILDER)
if spec is None or spec.loader is None:
    raise RuntimeError("v11 notebook builder missing")
v11 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v11)


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"expected one notebook anchor: {old[:70]!r}")
    return source.replace(old, new, 1)


def set_cell(notebook: dict, index: int, source: str) -> None:
    cell = notebook["cells"][index]
    if cell["cell_type"] == "code":
        compile(source, f"<v15 cell {index}>", "exec")
    cell["source"] = source.splitlines(keepends=True)
    cell["id"] = f"betatour_v15_{index:02d}_{hashlib.sha256(source.encode()).hexdigest()[:10]}"


def make_notebook() -> dict:
    notebook = v11.make_notebook()
    old = lambda index: "".join(notebook["cells"][index]["source"])

    set_cell(notebook, 0, """# BetaTour v15 — canlı aday değiştirme pilotu

LKH-3.0.14'ün POPMUSIC-6 aday listesiyle ilk denemesi tamamlandığında, en çok düğümlerin %5'inde düşük öncelikli bir adayı başka bir adayla **yerinde değiştiriyoruz**. BetaTour, en yakın yeni komşu ve deterministik rastgele seçim aynı kaynak düğümleri ve aynı çıkarılacak adayları kullanır. İlk altı turun bütün kenarları ve her aday satırının ilk iki girdisi korunur. Native POPMUSIC-6 kesintisiz pratik kıyastır; sham aynı duraklamayı ve model çıkarımını öder.

Bu **daha önce incelenmiş dört 10.000 düğümlü örnek üzerinde keşif pilotudur**. Sonuçlar bir üstünlük iddiası veya bağımsız doğrulama değildir. Bir politika gerçekten umut verirse, kodu ve analiz planını dondurup yeni örneklerde tekrar test edeceğiz. Her kol için 120 saniyelik toplam duvar saati sınırına ilk tur, bekleme, model çıkarımı ve sonraki arama dahildir. Bir örnekteki beş kol sırayla, her kolda altı LKH süreci paralel çalışır.

Colab Pro'da CPU, GPU veya High RAM çalışma ortamı seçebilirsiniz. CUDA varsa model onu kullanır; yoksa CPU kullanır. Özellikle 6'dan az vCPU, altı eşzamanlı LKH sürecini yavaşlatır; sonuçlar yine donanım profiliyle kaydedilir. Oturum kesilirse hücreleri yeniden çalıştırın: tamamlanan kollar `MyDrive/BetaTour/v15_live_swap_pilot` içinde doğrulanıp atlanır. Aynı Drive deneyi farklı GPU/vCPU profiliyle sürdürülmez. Pilot tamamlandığında sonuç ZIP'i tarayıcıya otomatik indirilir; ham denetim ZIP'i Drive'da kalır.

Yaklaşık süre: kurulum/derleme 3–8 dakika, 4 örnek × 5 kol × 120 saniye = 40 dakika çözücü, arşiv 2–5 dakika. CPU ve az vCPU ile ilk altı tur 120 saniyede tamamlanamazsa o kol eksik tanı olarak kaydedilir; sonuç karşılaştırmasına girmez.

İki vCPU'lu CPU ortamında bariyer 120 saniyede yetişmezse hücre 1'de dört `budget_s` değerini `300.0` ve `EXPERIMENT_NAME` değerini `v15_live_swap_pilot_cpu300` yapıp yeni bir Drive deneyi başlatın. Bu ayar beş kolu aynı 300 saniyelik bütçede karşılaştırır; 120 saniyelik GPU pilotuyla birleştirmeyin.
""")

    set_cell(notebook, 1, """EXPERIMENT_NAME = "v15_live_swap_pilot"
SEEDS = list(range(301, 307))
ARMS = ["native_pop6", "sham", "random_swap", "nearest_swap", "beta_swap"]
INSTANCE_SPECS = [
    ["uni10k", 10000, 2, "uniform", 4107, 120.0],
    ["clu10k", 10000, 2, "clustered", 4208, 120.0],
]
EXPECTED_XY_SHA256 = {
    "augment_uni10k_000": "0d5d680b465faa68d33467c4f70bb5ad1e52b25a2d444066ab922130458efcd6",
    "augment_uni10k_001": "00747175507524f481bd6e34ea55f0a5474d3dcd642a67543d871d9a64b53796",
    "augment_clu10k_000": "072fc96ec668bb9c95b3a52562f23cedb0b2e89494946d159243b638b7b3dc8e",
    "augment_clu10k_001": "965fe4315060118d313a357404f750a23ea89cc11c174cb1138a9f32b312ca22",
}
MAX_NEW_ARMS = None  # Örneğin 2: oturumda iki yeni kol çalıştırıp sonra devam edin.
SWAP_FRACTION = 0.05
GRAPH_K = 16
REPO_COMMIT = "aac51871a50df33daba2c0492cca2fe8f0c31089"
CHECKPOINT_SHA256 = "2322f66c752913f6179f8c61f6b50b0f9d21426857d3ec0155e02891fb3ebd70"
LKH_TAR_SHA256 = "4c4b9e6a77e50a1eb1eb2c8cb384ed5ef64e0dc88fdbf37bd693798469e1cf33"
LKH_FINDTOUR_SHA256 = "567e72acc9b7392488dd158be4df0267560424204a0e88d2d9b62f923a7c0eee"
assert len(SEEDS) == 6 and len(EXPECTED_XY_SHA256) == 4
""")

    set_cell(notebook, 2, """import base64, hashlib, importlib.util, json, math, os, platform, shutil, subprocess, sys, tarfile, time, urllib.request, uuid, zipfile
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
for package in ("scipy", "pandas"):
    if importlib.util.find_spec(package) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)
import numpy as np
import scipy
import pandas as pd
import torch
from IPython.display import display

torch.set_num_threads(1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "none"
CPU_COUNT = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
cpu_max = Path("/sys/fs/cgroup/cpu.max")
if cpu_max.exists():
    quota, period = cpu_max.read_text().split()[:2]
    if quota != "max":
        CPU_COUNT = min(CPU_COUNT, max(1, math.floor(int(quota) / int(period))))
CPU_COUNT = max(1, CPU_COUNT)
def cpu_model_name():
    info = Path("/proc/cpuinfo")
    if info.exists():
        for line in info.read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"

WORK_ROOT = Path("/content/betatour_v15_swap_work")
WORK_ROOT.mkdir(parents=True, exist_ok=True)
REPO = WORK_ROOT / "repo"
if not (REPO / ".git").exists():
    subprocess.run(["git", "clone", "--quiet", "--depth=1", "--filter=blob:none", "--no-checkout",
                    "https://github.com/kurbankotan/betatour-tsp-lkh-candidates.git", str(REPO)], check=True)
subprocess.run(["git", "-C", str(REPO), "sparse-checkout", "init", "--cone"], check=True)
subprocess.run(["git", "-C", str(REPO), "sparse-checkout", "set", "src", "checkpoints"], check=True)
subprocess.run(["git", "-C", str(REPO), "fetch", "--quiet", "--depth=1", "origin", REPO_COMMIT], check=True)
subprocess.run(["git", "-C", str(REPO), "checkout", "--quiet", "--detach", REPO_COMMIT], check=True)
assert subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip() == REPO_COMMIT
assert hashlib.sha256((REPO / "checkpoints/betatour_v2.pt").read_bytes()).hexdigest() == CHECKPOINT_SHA256
sys.path.insert(0, str(REPO / "src"))
import tsplab as T
import tsp_model as M
MODEL, _ = M.load_model(str(REPO / "checkpoints/betatour_v2.pt"), DEVICE)
assert MODEL.cfg.get("use_ensemble") is True

from google.colab import drive
drive.mount("/content/drive")
OUTPUT_ROOT = Path("/content/drive/MyDrive/BetaTour") / EXPERIMENT_NAME
for sub in ("instances", "code", "summary", "errors", "archives"):
    (OUTPUT_ROOT / sub).mkdir(parents=True, exist_ok=True)
HARDWARE_PROFILE = {"device": DEVICE.type, "gpu": GPU_NAME,
                    "vcpus": CPU_COUNT, "cpu_model": cpu_model_name()}
HARDWARE = {**HARDWARE_PROFILE,
            "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
            "cuda": torch.version.cuda}
if CPU_COUNT < 6:
    print("UYARI: Altı LKH süreci", CPU_COUNT, "vCPU paylaşacak; pilot yavaşlayabilir.")
print("Hazır:", HARDWARE, "Drive:", OUTPUT_ROOT)
""")

    files = {
        f"v15_live_swap/{name}": HERE / name for name in (
            "coordinator.py", "BetaTourSwapHook.h", "BetaTourVisit.h",
            "build_search_hook.py", "test_search_hook.py", "policy.py", "pilot.py", "run_cell.py",
            "analysis_cell.py", "README.md", "build_colab_notebook.py")
    }
    bundle = {name: base64.b64encode(path.read_bytes()).decode("ascii")
              for name, path in files.items()}
    set_cell(notebook, 3, """BUNDLED_FILES = """ + repr(bundle) + """
CODE_ROOT = WORK_ROOT / "code"
for rel, encoded in BUNDLED_FILES.items():
    content = base64.b64decode(encoded)
    for dest in (CODE_ROOT / rel, OUTPUT_ROOT / "code" / rel):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            assert dest.read_bytes() == content, f"Deney kodu değişmiş: {rel}"
        else:
            dest.write_bytes(content)
CODE_SHA256 = {rel: hashlib.sha256(base64.b64decode(encoded)).hexdigest()
               for rel, encoded in BUNDLED_FILES.items()}
sys.path.insert(0, str(CODE_ROOT / "v15_live_swap"))
import build_search_hook
from test_search_hook import run_smoke as run_swap_hook_smoke
from coordinator import read_tour
from pilot import run_guided_arm, run_native_arm, sha256_file
from policy import POLICY_VERSION
print("Dondurulmuş deney kodu:", CODE_SHA256)
""")

    build = old(4)
    build = replace_once(build, 'PATCHED_DIR = LKH_ROOT / "LKH-3.0.14-BetaTour-v11r2"',
                         'PATCHED_DIR = LKH_ROOT / "LKH-3.0.14-BetaTour-v15-swap"')
    build = replace_once(build, 'assert patch_manifest["search_hook_sha256"] == CODE_SHA256["v11_candidate/BetaTourSearchHook.h"]',
                         'assert patch_manifest["search_hook_sha256"] == CODE_SHA256["v15_live_swap/BetaTourSwapHook.h"]')
    build = replace_once(build, 'assert patch_manifest["visit_header_sha256"] == CODE_SHA256["v11_candidate/BetaTourVisit.h"]',
                         'assert patch_manifest["visit_header_sha256"] == CODE_SHA256["v15_live_swap/BetaTourVisit.h"]')
    build = replace_once(build, '"action_slot_rule": patch_manifest["action_slot_rule"],',
                         '"action_rule": patch_manifest["action_rule"],')
    build = replace_once(build, 'BUILD_INFO = {"native_sha256": sha256_file(NATIVE_LKH),',
                         'assert patch_manifest["swap_at_barrier"] is True\n'
                         'run_swap_hook_smoke(PATCHED_LKH)\n'
                         'BUILD_INFO = {"native_sha256": sha256_file(NATIVE_LKH),')
    build = replace_once(build, 'print("LKH 3.0.14 derlendi:", {k: v for k, v in BUILD_INFO.items() if k != "candidate_reordering"})',
                         'print("LKH 3.0.14 derlendi:", {k: v for k, v in BUILD_INFO.items() if k != "candidate_reordering"})')
    set_cell(notebook, 4, build)

    set_cell(notebook, 5, """def write_json_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
    os.replace(tmp, path)

MANIFEST = {"schema": 1, "profile": "four_reused_10k_development_instances",
            "experiment_name": EXPERIMENT_NAME, "arms": ARMS, "seeds": SEEDS,
            "instance_specs": INSTANCE_SPECS, "xy_hashes": EXPECTED_XY_SHA256,
            "policy_version": POLICY_VERSION, "swap_fraction": SWAP_FRACTION,
            "graph_k": GRAPH_K, "repo_commit": REPO_COMMIT,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "hardware_profile": HARDWARE_PROFILE, "code_sha256": CODE_SHA256,
            "build": BUILD_INFO,
            "analysis_plan": "Exploratory 4-instance swap pilot; best-of-six valid cost at a common 120-second wall deadline; no independent superiority claim."}
manifest_path = OUTPUT_ROOT / "manifest.json"
if manifest_path.exists():
    saved = json.loads(manifest_path.read_text())
    assert saved == MANIFEST, "Drive'daki pilot kodu, ayarı veya GPU/vCPU profili farklı; yeni deney adı kullanın."
else:
    write_json_atomic(manifest_path, MANIFEST)

# Exercise model inference, the patched first-trial barrier, and native parity.
warm_xy = T.to_int_coords(T.gen_instance(100, "uniform", np.random.default_rng(9090)))
warm_succ = np.stack([np.roll(np.arange(100), -1)] * 6, axis=1).astype(np.int32)
M.predict_tourfold(MODEL, warm_xy, GRAPH_K, DEVICE,
                   ens_succ=warm_succ, ens_cost=np.ones(6))
warm_problem = WORK_ROOT / "warmup.tsp"
T.write_tsp(warm_problem, warm_xy)
warm_dir = WORK_ROOT / "warmup" / uuid.uuid4().hex
warm_dir.parent.mkdir(exist_ok=True)
warm = run_guided_arm("sham", PATCHED_LKH, warm_problem, warm_xy, warm_dir,
                      SEEDS, max(45.0, 90.0 / CPU_COUNT), MODEL, DEVICE, T, M,
                      fraction=SWAP_FRACTION, graph_k=GRAPH_K,
                      decision_cache_dir=WORK_ROOT / "warmup_cache",
                      model_identity=CHECKPOINT_SHA256)
assert warm["policy_audit"]["selected_actions_per_seed"] == [0] * 6
def edge_set(tour):
    tour = np.asarray(tour, dtype=np.int32)
    return set(zip(np.minimum(tour, np.roll(tour, -1)),
                   np.maximum(tour, np.roll(tour, -1))))
for seed in SEEDS:
    native = T.run_lkh(str(NATIVE_LKH), str(warm_problem),
                       {"CANDIDATE_SET_TYPE": "POPMUSIC", "MAX_CANDIDATES": 6,
                        "SUBGRADIENT": "NO", "SEED": seed, "MAX_TRIALS": 1,
                        "TIME_LIMIT": 30, "STOP_AT_OPTIMUM": "NO"},
                       str(WORK_ROOT / "warmup" / f"native_{seed}_{uuid.uuid4().hex}"), timeout=30)
    first = np.asarray(read_tour(warm_dir / f"seed_{seed}" / "first.tour")) - 1
    assert edge_set(native["tour"]) == edge_set(first), f"İlk tur uyuşmuyor: {seed}"
print("Model, yama, korunan kenarlar ve ilk tur eşitliği doğrulandı.")
""")

    instances = old(6)
    set_cell(notebook, 6, instances)
    set_cell(notebook, 7, (HERE / "run_cell.py").read_text())
    set_cell(notebook, 8, (HERE / "analysis_cell.py").read_text())

    archive = old(9)
    archive = replace_once(archive, '"action.txt", "stdout.txt"',
                           '"action.txt", "protected.edges", "stdout.txt"')
    set_cell(notebook, 9, archive)
    notebook["metadata"]["colab"]["name"] = OUTPUT.name
    notebook["metadata"].pop("accelerator", None)
    return notebook


def main() -> None:
    notebook = make_notebook()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
