# Kurulum (3-5 dk): depo (sabit commit), protokolün GitHub denetimi, iki modülün denetimi, LKH-3.0.14, BetaTour.
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
import uuid
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
PARALLEL = max(1, CPU_COUNT // 2)          # protokol: görünen vCPU'ların yarısı kadar LKH süreci aynı anda
assert TEST_MODE or CPU_COUNT >= 8, f"En az 8 vCPU gerekli (Yüksek RAM seçin); görünen: {CPU_COUNT}"


def cpu_model_name():
    path = Path("/proc/cpuinfo")
    if path.exists():
        for line in path.read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


WORK_ROOT = Path(os.environ.get("BETATOUR_V30_WORK", "/content/betatour_v30_work"))
WORK_ROOT.mkdir(parents=True, exist_ok=True)

# Depo: run v27-v29 ile aynı sabit commit (örnek üreticisi, TSPLIB ve LKH yardımcıları, BetaTour ve model).
if TEST_MODE and os.environ.get("BETATOUR_V30_REPO"):
    REPO = Path(os.environ["BETATOUR_V30_REPO"])
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

# Protokol koşudan önce GitHub'da olmalı: main dalındaki protokol bu defterin kurulduğu protokolle bayt bayt aynı
# değilse ya da GitHub'daki defter bu kurulum değilse kurulum durur.
PROTOCOL = {"path": PROTOCOL_PATH, "sha256": PROTOCOL_SHA256, "github_head": None, "protocol_commit": None,
            "protocol_commit_time": None, "notebook_on_github": None, "checked_utc": None}
if REQUIRE_PUSHED_PROTOCOL:
    subprocess.run(["git", "-C", str(REPO), "fetch", "--quiet", "origin", "main"], check=True)
    github_head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "FETCH_HEAD"], text=True).strip()
    shown = subprocess.run(["git", "-C", str(REPO), "show", f"{github_head}:{PROTOCOL_PATH}"],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert shown.returncode == 0, \
        f"Protokol GitHub'da yok ({PROTOCOL_PATH}). Önce depoyu commit ve push edin, sonra defteri yeniden çalıştırın."
    assert hashlib.sha256(shown.stdout).hexdigest() == PROTOCOL_SHA256, \
        "GitHub'daki protokol bu defterin protokolüyle aynı değil. Protokol değiştiyse defter yeniden kurulmalıdır."
    pushed = subprocess.run(["git", "-C", str(REPO), "show", f"{github_head}:{NOTEBOOK_FILE}"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert pushed.returncode == 0 and NOTEBOOK_BUILD.encode() in pushed.stdout, \
        "GitHub'daki defter bu defterle aynı kurulum değil. Depodaki defteri push edip onu çalıştırın."
    last = subprocess.check_output(["git", "-C", str(REPO), "log", "-1", "--format=%H %cI", github_head, "--",
                                    PROTOCOL_PATH], text=True).split()
    PROTOCOL.update(github_head=github_head, protocol_commit=last[0], protocol_commit_time=last[1],
                    notebook_on_github=True, checked_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    print("Protokol GitHub'da:", PROTOCOL_PATH, "| commit", last[0][:10], last[1])


def checked_module(file_name, expected):
    """Önceki hücrelerin yazdığı modül; içerik depodakiyle bayt bayt aynı olmalıdır (dosya sonundaki satır sonu sayısı
    arayüze göre değişebildiği için tek satır sonuna indirilir)."""
    path = Path(file_name).resolve()
    assert path.exists(), f"Önce {file_name} dosyasını yazan hücreyi çalıştırın."
    data = path.read_bytes().rstrip(b"\n") + b"\n"
    digest = hashlib.sha256(data).hexdigest()
    assert digest == expected, f"{file_name} beklenen modülle aynı değil; o hücreyi değiştirmeden çalıştırın."
    if path.read_bytes() != data:
        path.write_bytes(data)
    return path, digest


ALPHAENC_PATH, _ = checked_module("alphaenc_v21.py", ALPHAENC_SHA256)
TTO_PATH, _ = checked_module("tto_v30.py", TTO_SHA256)
sys.path.insert(0, str(TTO_PATH.parent))
import alphaenc_v21 as A
import tto_v30 as V

if SAVE_TO_DRIVE:
    OUTPUT_ROOT = Path("/content/drive/MyDrive/BetaTour") / EXPERIMENT_NAME
else:
    OUTPUT_ROOT = WORK_ROOT / "results" / EXPERIMENT_NAME
for sub in ["instances", "runs", "summary", "code", "errors", "selfcheck"]:
    (OUTPUT_ROOT / sub).mkdir(parents=True, exist_ok=True)
shutil.copy2(ALPHAENC_PATH, OUTPUT_ROOT / "code" / "alphaenc_v21.py")
shutil.copy2(TTO_PATH, OUTPUT_ROOT / "code" / "tto_v30.py")


def run_build(cmd, cwd, what):
    build = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=1800)
    if build.returncode != 0:
        raise RuntimeError(what + " derlemesi başarısız:\n" + build.stdout[-3000:])


def get_lkh_3014():
    """Değiştirilmemiş LKH-3.0.14: resmi kaynak; ulaşılamazsa SHA256 denetimli sabit kopya (önceki koşulardaki gibi)."""
    if TEST_MODE and os.environ.get("BETATOUR_V30_LKH"):
        return os.environ["BETATOUR_V30_LKH"], "local test binary"
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert TEST_MODE or DEVICE.type == "cuda", "GPU gerekli (L4 ya da T4): BetaTour çıkarımı GPU'da yapılır."
checkpoint_path = REPO / "checkpoints" / "betatour_v2.pt"
assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == CHECKPOINT_SHA256, \
    "Model dosyası beklenen SHA256 değeriyle eşleşmiyor."
MODEL, CK = M.load_model(str(checkpoint_path), DEVICE)      # sabitlenmiş, güvenilen depo
assert MODEL.cfg.get("use_ensemble") is True

HARDWARE = {"device": str(DEVICE), "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "none",
            "vcpus_visible": CPU_COUNT, "parallel_lkh": PARALLEL, "cpu_model": cpu_model_name(),
            "ram_gb": T.env_info().get("ram_gb"), "python": platform.python_version(), "numpy": np.__version__,
            "torch": torch.__version__}
MANIFEST = {"schema": 1, "experiment_name": EXPERIMENT_NAME, "test_mode": TEST_MODE, "protocol": PROTOCOL,
            "code_version": V.VERSION, "alphaenc_version": A.VERSION, "alphaenc_sha256": ALPHAENC_SHA256,
            "tto_sha256": TTO_SHA256, "notebook_build": NOTEBOOK_BUILD, "repo_commit": REPO_COMMIT,
            "checkpoint_sha256": CHECKPOINT_SHA256, "lkh_source": LKH_SOURCE,
            "design": {"seeds": SEEDS, "arms": ARMS, "k_list": K_LIST, "k_union": K_UNION, "k_learned": K_LEARNED,
                       "k_graph": K_GRAPH, "cap_per_node": CAP_PER_NODE, "par_factor": PAR_FACTOR,
                       "tie_fraction": TIE_FRACTION,
                       "job_order_seed": JOB_ORDER_SEED, "synthetic": [list(x) for x in SYNTHETIC],
                       "optima": OPTIMA, "problem_sha256": PROBLEM_SHA256, "tsplib": TSPLIB_NAMES},
            "hardware": HARDWARE}
manifest_path = OUTPUT_ROOT / "manifest.json"
if manifest_path.exists():
    saved = json.loads(manifest_path.read_text())
    for key in ["schema", "design", "alphaenc_sha256", "tto_sha256", "notebook_build", "repo_commit",
                "checkpoint_sha256"]:
        assert saved[key] == MANIFEST[key], f"{key} değişti. Ayrı bir EXPERIMENT_NAME kullanın."
    same = (saved["hardware"]["cpu_model"] == HARDWARE["cpu_model"]
            and saved["hardware"]["vcpus_visible"] == HARDWARE["vcpus_visible"])
    assert ALLOW_HARDWARE_CHANGE or same, \
        (f"Makine türü değişti ({saved['hardware']['cpu_model']}, {saved['hardware']['vcpus_visible']} vCPU -> "
         f"{HARDWARE['cpu_model']}, {HARDWARE['vcpus_visible']} vCPU). Aynı türü seçin; bulunamıyorsa ilk hücrede "
         "ALLOW_HARDWARE_CHANGE = True yapın (rapora not düşülür).")
    if saved["hardware"] != HARDWARE:
        print("UYARI: donanım veya kütüphane sürümleri ilk oturumdan farklı; her koşunun donanımı ayrıca kaydedilir.")
else:
    manifest_path.write_text(json.dumps(MANIFEST, indent=2, ensure_ascii=False))

print(f"Hazır ({time.time() - t_setup:.0f} s): {CPU_COUNT} vCPU, aynı anda {PARALLEL} LKH süreci | "
      f"{HARDWARE['cpu_model']} | {HARDWARE['gpu']} | LKH: {LKH_SOURCE} | modüller {V.VERSION}, {A.VERSION} | "
      f"sonuçlar: {OUTPUT_ROOT}")
