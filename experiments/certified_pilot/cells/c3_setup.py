# Kurulum (5-8 dk): depo (sabit commit), certpilot.py denetimi, LKH-3.0.14, Concorde + QSopt (resmi kaynak), BetaTour.
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import torch

CPU_COUNT = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
cpu_max = Path("/sys/fs/cgroup/cpu.max")
if cpu_max.exists():
    quota, period = cpu_max.read_text().split()[:2]
    if quota != "max":
        CPU_COUNT = min(CPU_COUNT, max(1, math.floor(int(quota) / int(period))))
if CPU_COUNT < 8 and not TEST_MODE:
    print(f"UYARI: {CPU_COUNT} vCPU görünüyor; Concorde adımı uzun sürer. Yüksek RAM'li L4 ortamı (12 vCPU) önerilir.")


def cpu_model_name():
    path = Path("/proc/cpuinfo")
    if path.exists():
        for line in path.read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


WORK_ROOT = Path(os.environ.get("CERTPILOT_WORK", "/content/certpilot_work"))
WORK_ROOT.mkdir(parents=True, exist_ok=True)

# Depo: run v27-v29 ile aynı sabit commit (örnek üreticisi ve LKH yardımcıları: tsplab; BetaTour: tsp_model ve model).
if TEST_MODE and os.environ.get("CERTPILOT_REPO"):
    REPO = Path(os.environ["CERTPILOT_REPO"])
else:
    REPO = WORK_ROOT / "repo"
    if not (REPO / ".git").exists():
        subprocess.run(["git", "clone", "--quiet", "https://github.com/kurbankotan/betatour-tsp-lkh-candidates.git",
                        str(REPO)], check=True)
    subprocess.run(["git", "-C", str(REPO), "fetch", "--quiet", "origin", REPO_COMMIT], check=True)
    subprocess.run(["git", "-C", str(REPO), "checkout", "--quiet", REPO_COMMIT], check=True)
    assert subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip() == REPO_COMMIT
sys.path.insert(0, str(REPO / "src"))
import tsplab as T
import tsp_model as M


def checked_module(file_name, expected):
    """Önceki hücrenin yazdığı modül; içerik depodaki certpilot.py ile bayt bayt aynı olmalıdır (dosya sonundaki satır
    sonu sayısı arayüze göre değişebildiği için tek satır sonuna indirilir)."""
    path = Path(file_name).resolve()
    assert path.exists(), f"Önce {file_name} dosyasını yazan hücreyi çalıştırın."
    data = path.read_bytes().rstrip(b"\n") + b"\n"
    digest = hashlib.sha256(data).hexdigest()
    assert digest == expected, f"{file_name} beklenen modülle aynı değil; o hücreyi değiştirmeden çalıştırın."
    if path.read_bytes() != data:
        path.write_bytes(data)
    return path, digest


CODE_PATH, CODE_SHA256 = checked_module("certpilot.py", CODE_SHA256_EXPECTED)
sys.path.insert(0, str(CODE_PATH.parent))
import certpilot as C

if SAVE_TO_DRIVE:
    OUTPUT_ROOT = Path("/content/drive/MyDrive/BetaTour") / EXPERIMENT_NAME
else:
    OUTPUT_ROOT = WORK_ROOT / "results" / EXPERIMENT_NAME
for sub in ["instances", "logs", "summary", "code"]:
    (OUTPUT_ROOT / sub).mkdir(parents=True, exist_ok=True)
shutil.copy2(CODE_PATH, OUTPUT_ROOT / "code" / "certpilot.py")


def run_build(cmd, cwd, what):
    build = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=1800)
    if build.returncode != 0:
        raise RuntimeError(what + " derlemesi başarısız:\n" + build.stdout[-3000:])


def get_lkh_3014():
    """Değiştirilmemiş LKH-3.0.14: resmi kaynak; ulaşılamazsa SHA256 denetimli sabit kopya (önceki koşulardaki gibi)."""
    if TEST_MODE and os.environ.get("CERTPILOT_LKH"):
        return os.environ["CERTPILOT_LKH"], "local test binary"
    root = WORK_ROOT / "lkh"
    binary = root / "LKH-3.0.14" / "LKH"
    marker = root / "source.txt"
    if binary.exists() and marker.exists():
        return str(binary), marker.read_text().strip()
    root.mkdir(parents=True, exist_ok=True)
    options = [("official", "https://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.0.14.tgz", None, 12),
               ("pinned copy", "https://raw.githubusercontent.com/ZhekaiChen/ALCO/main/third_party/LKH3/LKH-3.0.14.tar",
                "4c4b9e6a77e50a1eb1eb2c8cb384ed5ef64e0dc88fdbf37bd693798469e1cf33", 90)]
    last_error = None
    for label, url, expected_sha, timeout in options:
        archive = root / ("LKH-3.0.14.tar" if url.endswith(".tar") else "LKH-3.0.14.tgz")
        try:
            print("LKH kaynağı:", label, url)
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response, archive.open("wb") as f:
                shutil.copyfileobj(response, f)
            if expected_sha:
                got = hashlib.sha256(archive.read_bytes()).hexdigest()
                assert got == expected_sha, f"LKH kopyası SHA256 uyuşmazlığı: {got}"
            shutil.rmtree(root / "LKH-3.0.14", ignore_errors=True)
            with tarfile.open(archive, "r:*") as tar:
                try:
                    tar.extractall(root, filter="data")
                except TypeError:
                    tar.extractall(root)
            run_build(["make", "-j", str(min(4, CPU_COUNT))], root / "LKH-3.0.14", "LKH")
            assert binary.exists(), "LKH ikilisi oluşmadı"
            marker.write_text(f"{label}: {url}")
            return str(binary), f"{label}: {url}"
        except Exception as exc:
            last_error = exc
            print(f"{label} başarısız: {exc}")
    raise RuntimeError(f"LKH 3.0.14 indirilemedi veya derlenemedi: {last_error}")


t_setup = time.time()
LKH, LKH_SOURCE = get_lkh_3014()
if TEST_MODE and os.environ.get("CERTPILOT_CONCORDE"):
    CONCORDE_INFO = {"binary": os.environ["CERTPILOT_CONCORDE"], "sources": "local test stand-in (not Concorde)"}
else:
    print("Concorde ve QSopt indiriliyor ve derleniyor (2-4 dk)...")
    CONCORDE_INFO = C.build_concorde(WORK_ROOT / "concorde_build", jobs=min(8, CPU_COUNT))
CONCORDE = CONCORDE_INFO["binary"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = REPO / "checkpoints" / "betatour_v2.pt"
assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == CHECKPOINT_SHA256, \
    "Model dosyası beklenen SHA256 değeriyle eşleşmiyor."
MODEL, CK = M.load_model(str(checkpoint_path), DEVICE)      # sabitlenmiş, güvenilen depo
assert MODEL.cfg.get("use_ensemble") is True

HARDWARE = {"device": str(DEVICE), "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "none",
            "vcpus_visible": CPU_COUNT, "cpu_model": cpu_model_name(), "ram_gb": T.env_info().get("ram_gb"),
            "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}
MANIFEST = {"schema": 1, "experiment_name": EXPERIMENT_NAME, "test_mode": TEST_MODE, "code_version": C.VERSION,
            "code_sha256": CODE_SHA256, "notebook_build": NOTEBOOK_BUILD, "repo_commit": REPO_COMMIT,
            "checkpoint_sha256": CHECKPOINT_SHA256, "lkh_source": LKH_SOURCE,
            "concorde": {k: v for k, v in CONCORDE_INFO.items() if k != "binary"},
            "design": {"synthetic": [list(x) for x in SYNTHETIC], "tsplib": TSPLIB_NAMES,
                       "first_trial_seeds": FIRST_TRIAL_SEEDS, "k_list": K_LIST, "k_graph": K_GRAPH,
                       "rank_depth": RANK_DEPTH, "ub_time_per_node": UB_TIME_PER_NODE, "k_rc": K_RC,
                       "k_rc_self": K_RC_SELF,
                       "rc_tol": RC_TOL, "concorde_seed": CONCORDE_SEED, "root_timeout_s": ROOT_TIMEOUT_S,
                       "full_timeout_s": FULL_TIMEOUT_S, "self_n": SELF_N,
                       "decision": {"g2_max_mean_degree": G2_MAX_MEAN_DEGREE, "g3_margin_pp": G3_MARGIN_PP}},
            "hardware": HARDWARE}
manifest_path = OUTPUT_ROOT / "manifest.json"
if manifest_path.exists():
    saved = json.loads(manifest_path.read_text())
    for key in ["schema", "design", "code_sha256", "repo_commit", "checkpoint_sha256"]:
        assert saved[key] == MANIFEST[key], f"{key} değişti. Ayrı bir EXPERIMENT_NAME kullanın."
    if saved["hardware"] != HARDWARE:
        print("UYARI: donanım ilk oturumdan farklı; süre ölçümleri oturumlar arasında karşılaştırılamaz.")
else:
    manifest_path.write_text(json.dumps(MANIFEST, indent=2, ensure_ascii=False))

print(f"Hazır ({time.time() - t_setup:.0f} s): {CPU_COUNT} vCPU | {HARDWARE['cpu_model']} | aygıt: {DEVICE} | LKH: "
      f"{LKH_SOURCE} | Concorde: {CONCORDE} | kod {C.VERSION} {CODE_SHA256[:12]} | sonuçlar: {OUTPUT_ROOT}")
