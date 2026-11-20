import hashlib
import importlib.util
import json
import math
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

for package, module in [("scipy", "scipy"), ("pandas", "pandas")]:
    if importlib.util.find_spec(module) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import scipy
import torch

torch.set_num_threads(1)
if TEST_MODE:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    assert torch.cuda.is_available(), "Colab'da GPU çalışma ortamı seçin: L4 veya A100."
    DEVICE = torch.device("cuda")
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu"
CPU_COUNT = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
cpu_max = Path("/sys/fs/cgroup/cpu.max")
if cpu_max.exists():
    quota, period = cpu_max.read_text().split()[:2]
    if quota != "max":
        CPU_COUNT = min(CPU_COUNT, max(1, math.floor(int(quota) / int(period))))
assert CPU_COUNT >= W, f"{W} LKH süreci için en az {W} vCPU gerekli; görünen: {CPU_COUNT}"
if REQUIRE_12_VCPUS:
    assert CPU_COUNT >= 2 * W, f"Makaledeki düzen için 12 vCPU gerekli (Yüksek RAM seçin); görünen: {CPU_COUNT}"


def cpu_model_name():
    path = Path("/proc/cpuinfo")
    if path.exists():
        for line in path.read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


CPU_MODEL = cpu_model_name()
WORK_ROOT = Path(os.environ.get("BETATOUR_V22_WORK", "/content/betatour_v22_work"))
WORK_ROOT.mkdir(parents=True, exist_ok=True)

# Depo: run v5-v21 ile aynı sabit commit (model kodu ve eğitilmiş model).
if TEST_MODE and os.environ.get("BETATOUR_V22_REPO"):
    REPO = Path(os.environ["BETATOUR_V22_REPO"])
else:
    REPO = WORK_ROOT / "repo"
    if not (REPO / ".git").exists():
        subprocess.run(["git", "clone", "--quiet",
                        "https://github.com/kurbankotan/betatour-tsp-lkh-candidates.git", str(REPO)], check=True)
    subprocess.run(["git", "-C", str(REPO), "fetch", "--quiet", "origin", REPO_COMMIT], check=True)
    subprocess.run(["git", "-C", str(REPO), "checkout", "--quiet", REPO_COMMIT], check=True)
    assert subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip() == REPO_COMMIT
sys.path.insert(0, str(REPO / "src"))
import tsplab as T
import tsp_model as M

# Yardımcı modül (önceki hücrenin yazdığı dosya; run v21 ile bayt bayt aynı). Dosya sonundaki satır sonu sayısı arayüze göre
# değişebildiği için tek satır sonuna indirilir; içerik depodaki modülle bayt bayt aynı olmalıdır.
CODE_PATH = Path("alphaenc_v21.py").resolve()
assert CODE_PATH.exists(), "Önce bir önceki hücreyi (alphaenc_v21.py) çalıştırın."
code_bytes = CODE_PATH.read_bytes().rstrip(b"\n") + b"\n"
CODE_SHA256 = hashlib.sha256(code_bytes).hexdigest()
assert CODE_SHA256 == CODE_SHA256_EXPECTED, "alphaenc_v21.py depodaki modülle aynı değil; önceki hücreyi değiştirmeden çalıştırın."
if CODE_PATH.read_bytes() != code_bytes:
    CODE_PATH.write_bytes(code_bytes)
sys.path.insert(0, str(CODE_PATH.parent))
import alphaenc_v21 as A

checkpoint_path = REPO / "checkpoints" / "betatour_v2.pt"
assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == CHECKPOINT_SHA256, \
    "Model dosyası beklenen SHA256 değeriyle eşleşmiyor."
MODEL, CK = M.load_model(str(checkpoint_path), DEVICE)   # sabitlenmiş, güvenilen depo
assert MODEL.cfg.get("use_ensemble") is True

if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    OUTPUT_ROOT = Path("/content/drive/MyDrive/BetaTour") / EXPERIMENT_NAME
else:
    OUTPUT_ROOT = WORK_ROOT / "results" / EXPERIMENT_NAME
for sub in ["instances", "stages", "candidates", "errors", "summary", "code"]:
    (OUTPUT_ROOT / sub).mkdir(parents=True, exist_ok=True)
shutil.copy2(CODE_PATH, OUTPUT_ROOT / "code" / "alphaenc_v21.py")


def get_lkh_3014():
    """Resmi kaynak; ulaşılamazsa SHA256 denetimli sabit kopya (run v10-v12 ile aynı)."""
    if TEST_MODE and os.environ.get("BETATOUR_V22_LKH"):
        return os.environ["BETATOUR_V22_LKH"], "local test binary"
    root = WORK_ROOT / "lkh"
    binary = root / "LKH-3.0.14" / "LKH"
    marker = root / "source.txt"
    if binary.exists() and marker.exists():
        return str(binary), marker.read_text().strip()
    root.mkdir(parents=True, exist_ok=True)
    options = [
        ("official", "https://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.0.14.tgz", None, 12),
        ("pinned copy", "https://raw.githubusercontent.com/ZhekaiChen/ALCO/main/third_party/LKH3/LKH-3.0.14.tar",
         "4c4b9e6a77e50a1eb1eb2c8cb384ed5ef64e0dc88fdbf37bd693798469e1cf33", 90),
    ]
    last_error = None
    for label, url, expected_sha, timeout in options:
        archive = root / ("LKH-3.0.14.tar" if url.endswith(".tar") else "LKH-3.0.14.tgz")
        try:
            print("LKH kaynağı:", label, url)
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response, archive.open("wb") as f:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    f.write(block)
            if expected_sha:
                got = hashlib.sha256(archive.read_bytes()).hexdigest()
                assert got == expected_sha, f"LKH kopyası SHA256 uyuşmazlığı: {got}"
            shutil.rmtree(root / "LKH-3.0.14", ignore_errors=True)
            with tarfile.open(archive, "r:*") as tar:
                try:
                    tar.extractall(root, filter="data")
                except TypeError:
                    tar.extractall(root)
            assert (root / "LKH-3.0.14" / "Makefile").exists(), "LKH kaynak klasörü eksik"
            build = subprocess.run(["make", "-j", str(min(4, CPU_COUNT))], cwd=root / "LKH-3.0.14",
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=900)
            if build.returncode != 0:
                raise RuntimeError("LKH derlemesi başarısız:\n" + build.stdout[-3000:])
            assert binary.exists(), "LKH ikilisi oluşmadı"
            marker.write_text(f"{label}: {url}")
            return str(binary), f"{label}: {url}"
        except Exception as exc:
            last_error = exc
            print(f"{label} başarısız: {exc}")
    raise RuntimeError(f"LKH 3.0.14 indirilemedi veya derlenemedi: {last_error}")


LKH, LKH_SOURCE = get_lkh_3014()
# Deneyin konusu olan iki kural derlenen kaynakta var mı? (ChooseInitialTour.c, C durumu; LKH.h, Near makrosu)
walk_source = Path(LKH).parent / "SRC" / "ChooseInitialTour.c"
near_source = Path(LKH).parent / "SRC" / "INCLUDE" / "LKH.h"
WALK_SHA256 = None
if walk_source.exists() and near_source.exists():
    assert "NN->Alpha == 0 && (InBestTour(N, NextN) ||" in walk_source.read_text(errors="replace"), \
        "ChooseInitialTour.c beklenen kuralı içermiyor"
    assert "(a)->BestSuc ? InBestTour(a, b) : (a)->Dad == (b) || (b)->Dad == (a)" in near_source.read_text(errors="replace"), \
        "LKH.h beklenen Near makrosunu içermiyor"
    WALK_SHA256 = hashlib.sha256(walk_source.read_bytes()).hexdigest()

HARDWARE = {
    "gpu": GPU_NAME,
    "vcpus_visible": CPU_COUNT,
    "cpu_model": CPU_MODEL,
    "ram_gb": T.env_info().get("ram_gb"),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "numpy": np.__version__,
    "scipy": scipy.__version__,
}
MANIFEST = {
    "schema": 1,
    "experiment_name": EXPERIMENT_NAME,
    "test_mode": TEST_MODE,
    "design": {"synthetic": [list(x) for x in SYNTHETIC], "tsplib": TSPLIB_NAMES,
               "tsplib_budget": "max(10, 0.02 n) s", "pad_seconds": PAD_SECONDS},
    "seeds": SEEDS,
    "workers": W,
    "graph_k": K_GRAPH,
    "candidate_k": K_LIST,
    "learned_k": K_LEARNED,
    "arms": ARM_NAMES,
    "repo_commit": REPO_COMMIT,
    "checkpoint_sha256": CHECKPOINT_SHA256,
    "code_version": A.VERSION,
    "code_sha256": CODE_SHA256,
    "notebook_build": NOTEBOOK_BUILD,
    "lkh_source": LKH_SOURCE,
    "choose_initial_tour_sha256": WALK_SHA256,
    "hardware": HARDWARE,
}
manifest_path = OUTPUT_ROOT / "manifest.json"
if manifest_path.exists():
    saved = json.loads(manifest_path.read_text())
    for key in ["schema", "design", "seeds", "workers", "graph_k", "candidate_k", "learned_k", "arms",
                "repo_commit", "checkpoint_sha256", "code_version", "code_sha256", "notebook_build"]:
        assert saved[key] == MANIFEST[key], f"{key} değişti. Ayrı bir EXPERIMENT_NAME kullanın."
    assert saved["hardware"]["vcpus_visible"] == HARDWARE["vcpus_visible"], "vCPU sayısı değişti; aynı ortam türünü seçin."
    assert ALLOW_HARDWARE_CHANGE or saved["hardware"]["gpu"] == HARDWARE["gpu"], \
        (f"GPU türü değişti ({saved['hardware']['gpu']} -> {HARDWARE['gpu']}). Aynı türü seçin; bulunamıyorsa ilk "
         "hücrede ALLOW_HARDWARE_CHANGE = True yapın (her örneğin bütün kolları yine tek makinede koşar).")
    if saved["hardware"] != HARDWARE:
        print("UYARI: donanım veya kütüphane sürümleri ilk oturumdan farklı; her örneğin donanımı ayrıca kaydedilir.")
else:
    manifest_path.write_text(json.dumps(MANIFEST, indent=2, ensure_ascii=False))

# Aynı Colab sanal makinesinde defter yeniden çalıştırılırsa bu işaret korunur. Yeni makinede yarım kalan
# örnek baştan çalıştırılır; böylece bir örneğin bütün kolları tek makinede koşar.
session_path = WORK_ROOT / "runtime_session_id.txt"
if session_path.exists():
    SESSION_ID = session_path.read_text().strip()
else:
    SESSION_ID = uuid.uuid4().hex
    session_path.write_text(SESSION_ID)

print("Hazır:", GPU_NAME, "|", CPU_COUNT, "vCPU |", sum(p.numel() for p in MODEL.parameters()), "model parametresi |",
      "LKH:", LKH_SOURCE, "| kod", A.VERSION, CODE_SHA256[:12], "| sonuçlar:", OUTPUT_ROOT)
