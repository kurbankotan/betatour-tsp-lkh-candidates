import hashlib
import importlib.util
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if importlib.util.find_spec("pandas") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pandas"], check=True)
try:
    import pyscipopt
    assert pyscipopt.__version__ == PYSCIPOPT_VERSION
except (ImportError, AssertionError):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"pyscipopt=={PYSCIPOPT_VERSION}"], check=True)
    importlib.invalidate_caches()
    for loaded in [m for m in sys.modules if m == "pyscipopt" or m.startswith("pyscipopt.")]:
        del sys.modules[loaded]
    import pyscipopt
    assert pyscipopt.__version__ == PYSCIPOPT_VERSION, \
        f"PySCIPOpt {PYSCIPOPT_VERSION} kuruldu ama eski sürüm yüklü kaldı; Çalışma zamanı > Oturumu yeniden başlat deyip yeniden çalıştırın."
SCIP_VERSION = str(pyscipopt.Model().version())

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
assert CPU_COUNT >= W, f"{W} süreç için en az {W} vCPU gerekli; görünen: {CPU_COUNT}"
if REQUIRE_12_VCPUS:
    assert CPU_COUNT >= 2 * W, f"Önceki koşulardaki düzen için 12 vCPU gerekli (Yüksek RAM seçin); görünen: {CPU_COUNT}"


def cpu_model_name():
    path = Path("/proc/cpuinfo")
    if path.exists():
        for line in path.read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def gpu_name():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=30)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else "none"
    except Exception:
        return "none"


CPU_MODEL = cpu_model_name()
GPU_NAME = gpu_name()
WORK_ROOT = Path(os.environ.get("BETATOUR_V27_WORK", "/content/betatour_v27_work"))
WORK_ROOT.mkdir(parents=True, exist_ok=True)

# Depo: run v5-v26 ile aynı sabit commit (örnek üreticisi, BetaTour modeli ve eğitilmiş model dosyası).
if TEST_MODE and os.environ.get("BETATOUR_V27_REPO"):
    REPO = Path(os.environ["BETATOUR_V27_REPO"])
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

# Protokol koşudan önce GitHub'da olmalı: main dalındaki protokol dosyası bu defterin kurulduğu protokolle bayt bayt
# aynı değilse kurulum durur. Dosyayı son değiştiren commit ve zamanı manifeste yazılır (ilk örnekten önce).
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
    notebook_file = "notebooks/29_BetaTour_v27_union_exact_pilot.ipynb"
    pushed = subprocess.run(["git", "-C", str(REPO), "show", f"{github_head}:{notebook_file}"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert pushed.returncode == 0 and NOTEBOOK_BUILD.encode() in pushed.stdout, \
        "GitHub'daki defter bu defterle aynı kurulum değil. Depodaki defteri push edip onu çalıştırın."
    last = subprocess.check_output(["git", "-C", str(REPO), "log", "-1", "--format=%H %cI", github_head, "--",
                                    PROTOCOL_PATH], text=True).split()
    PROTOCOL.update(github_head=github_head, protocol_commit=last[0], protocol_commit_time=last[1],
                    notebook_on_github=True, checked_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    print("Protokol GitHub'da:", PROTOCOL_PATH, "| commit", last[0][:10], last[1])

RAM_GB = T.env_info().get("ram_gb")
assert RAM_GB is None or RAM_GB >= MIN_RAM_GB, \
    f"Yüksek RAM gerekli (GA-EAX süreçleri için): görünen {RAM_GB} GB, gereken en az {MIN_RAM_GB} GB"

# Bu deneyin iki yardımcı modülü (önceki iki hücrenin yazdığı dosyalar): run v26'nın modülü (değiştirilmeden) ve
# birleşim grafında tam çözüm modülü. Dosya sonundaki satır sonu sayısı arayüze göre değişebildiği için tek satır
# sonuna indirilir; içerik depodaki modüllerle bayt bayt aynı olmalıdır.
def checked_module(file_name, expected):
    path = Path(file_name).resolve()
    assert path.exists(), f"Önce {file_name} dosyasını yazan hücreyi çalıştırın."
    data = path.read_bytes().rstrip(b"\n") + b"\n"
    digest = hashlib.sha256(data).hexdigest()
    assert digest == expected, f"{file_name} beklenen modülle aynı değil; o hücreyi değiştirmeden çalıştırın."
    if path.read_bytes() != data:
        path.write_bytes(data)
    return path, digest


CODE_PATH, CODE_SHA256 = checked_module("byprod_v26.py", CODE_SHA256_EXPECTED)
UNION_PATH, UNION_SHA256 = checked_module("unionx_v27.py", UNION_SHA256_EXPECTED)
sys.path.insert(0, str(CODE_PATH.parent))
import byprod_v26 as B
import unionx_v27 as U

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert TEST_MODE or DEVICE.type == "cuda", "GPU gerekli (L4): BetaTour çıkarımı GPU'da yapılır."
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
for sub in ["instances", "tours", "stages", "errors", "summary", "code"]:
    (OUTPUT_ROOT / sub).mkdir(parents=True, exist_ok=True)
shutil.copy2(CODE_PATH, OUTPUT_ROOT / "code" / "byprod_v26.py")
shutil.copy2(UNION_PATH, OUTPUT_ROOT / "code" / "unionx_v27.py")


def run_build(cmd, cwd, what):
    build = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=1800)
    if build.returncode != 0:
        raise RuntimeError(what + " derlemesi başarısız:\n" + build.stdout[-3000:])


def findtour_matches(folder):
    path = Path(folder) / "SRC" / "FindTour.c"
    return path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == B.LKH_FINDTOUR_SHA256


def get_lkh_3014():
    """Değiştirilmemiş LKH-3.0.14. Resmi kaynak; ulaşılamazsa ya da FindTour.c beklenen dosya değilse SHA256
    denetimli sabit kopya (run v21-v23 ve v25 bu kopyayla koştu)."""
    if TEST_MODE and os.environ.get("BETATOUR_V27_LKH"):
        assert findtour_matches(Path(os.environ["BETATOUR_V27_LKH"]).parent), "yerel LKH kaynağı 3.0.14 değil"
        return os.environ["BETATOUR_V27_LKH"], "local test binary"
    root = WORK_ROOT / "lkh"
    binary = root / "LKH-3.0.14" / "LKH"
    marker = root / "source.txt"
    if binary.exists() and marker.exists() and findtour_matches(root / "LKH-3.0.14"):
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
            assert findtour_matches(root / "LKH-3.0.14"), "FindTour.c beklenen dosya değil (kayıt yaması uygulanamaz)"
            run_build(["make", "-j", str(min(4, CPU_COUNT))], root / "LKH-3.0.14", "LKH")
            assert binary.exists(), "LKH ikilisi oluşmadı"
            marker.write_text(f"{label}: {url}")
            return str(binary), f"{label}: {url}"
        except Exception as exc:
            last_error = exc
            print(f"{label} başarısız: {exc}")
    raise RuntimeError(f"LKH 3.0.14 indirilemedi veya derlenemedi: {last_error}")


def build_recording_lkh(plain_binary):
    """LKH'nin kayıt eden kopyası: FindTour.c'ye iki çağrı ve bir işlev eklenir (byprod_v26.patch_lkh_findtour).
    Arama durumu değişmez; öz sınama bunu bu makinede denetler. BETATOUR_BEST_ONLY ortam değişkeni verilirse
    yalnızca yeni en iyi turlar yazılır (LKH'nin kendi tur birleştirmesini yapan süreçlerde kullanılır)."""
    source = Path(plain_binary).parent
    dest = WORK_ROOT / "lkh_rec" / "LKH-3.0.14"
    binary, marker = dest / "LKH", dest / "patched.txt"
    if binary.exists() and marker.exists() and marker.read_text() == B.VERSION:
        return str(binary)
    shutil.rmtree(dest.parent, ignore_errors=True)
    shutil.copytree(source, dest, ignore=shutil.ignore_patterns("OBJ", "LKH", "*.o", "*.tgz", "*.tar"))
    (dest / "SRC" / "OBJ").mkdir(exist_ok=True)
    path = dest / "SRC" / "FindTour.c"
    path.write_text(B.patch_lkh_findtour(path.read_text()))
    run_build(["make", "-j", str(min(4, CPU_COUNT))], dest, "kayıt eden LKH")
    assert binary.exists(), "kayıt eden LKH ikilisi oluşmadı"
    marker.write_text(B.VERSION)
    return str(binary)


def get_eax():
    """GA-EAX (Nagata ve Kobayashi, 2013): yazarın GitHub deposu, sabit commit, MIT lisansı. İki sürüm de
    (Normal, Large) iki yamayla derlenir: env.cpp'ye zaman damgası ve duvar saati sınırı (patch_eax_env),
    evaluator.cpp'ye en yakın şehir listelerinin ızgarayla bulunması (patch_eax_evaluator; aynı listeler, 50.000
    düğümde yaklaşık bir dakika yerine bir saniyeden az). Öz sınama için ikinci yamasız birer kopya da derlenir."""
    if TEST_MODE and os.environ.get("BETATOUR_V27_EAX_SRC"):
        source, label = Path(os.environ["BETATOUR_V27_EAX_SRC"]), "local test clone"
    else:
        source = WORK_ROOT / "ga_eax_src"
        if not (source / ".git").exists():
            shutil.rmtree(source, ignore_errors=True)
            subprocess.run(["git", "clone", "--quiet", "https://github.com/nagata-yuichi/GA-EAX.git", str(source)],
                           check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--quiet", B.EAX_COMMIT], check=True)
        assert subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() == B.EAX_COMMIT
        label = "https://github.com/nagata-yuichi/GA-EAX @ " + B.EAX_COMMIT
    binaries, plain = {}, {}
    for version in ("Normal", "Large"):
        folder = source / "GA_EAX_1.0" / version
        assert B.source_digest(str(folder)) == B.EAX_TREE_SHA256[version], \
            f"GA-EAX kaynağı ({version}) sabitlenen commit ile aynı değil"
        for grid, target in ((True, binaries), (False, plain)):
            dest = WORK_ROOT / "ga_eax_build" / (version.lower() + ("" if grid else "_plain"))
            binary, marker = dest / "eax", dest / "patched.txt"
            if not (binary.exists() and marker.exists() and marker.read_text() == B.VERSION):
                shutil.rmtree(dest, ignore_errors=True)
                dest.mkdir(parents=True)
                for path in folder.iterdir():
                    if path.suffix in {".cpp", ".h"}:
                        shutil.copy2(path, dest / path.name)
                env_file = dest / "env.cpp"
                env_file.write_text(B.patch_eax_env(env_file.read_text(), version))
                if grid:
                    evaluator_file = dest / "evaluator.cpp"
                    evaluator_file.write_text(B.patch_eax_evaluator(evaluator_file.read_text(), version))
                run_build(["g++", "-O3", "-w", "-o", "eax"] + B.EAX_SOURCES + ["-lm"], dest, "GA-EAX " + version)
                assert binary.exists(), "GA-EAX ikilisi oluşmadı"
                marker.write_text(B.VERSION)
            target[version] = str(binary)
    return binaries, plain, label


LKH, LKH_SOURCE = get_lkh_3014()
LKH_REC = build_recording_lkh(LKH)
EAX, EAX_PLAIN, EAX_SOURCE = get_eax()

HARDWARE = {
    "gpu": GPU_NAME,
    "vcpus_visible": CPU_COUNT,
    "cpu_model": CPU_MODEL,
    "ram_gb": RAM_GB,
    "python": platform.python_version(),
    "numpy": np.__version__,
    "torch": torch.__version__,
    "pyscipopt": pyscipopt.__version__,
    "scip": SCIP_VERSION,
}
MANIFEST = {
    "schema": 1,
    "experiment_name": EXPERIMENT_NAME,
    "test_mode": TEST_MODE,
    "design": {"synthetic": [list(x) for x in SYNTHETIC], "decision_groups": DECISION_GROUPS,
               "candidate_k": K_LIST, "graph_k": K_GRAPH, "split": round(float(SPLIT), 6), "per_process": PER_PROCESS,
               "extra_fraction": EXTRA_FRACTION, "merge_seeds": MERGE_SEEDS, "eax_kids": EAX_KIDS,
               "eax_normal_max_n": EAX_NORMAL_MAX_N, "stop_pad_s": STOP_PAD_S, "exact_pad_s": EXACT_PAD_S},
    "protocol": PROTOCOL,
    "seeds": SEEDS,
    "workers": W,
    "repo_commit": REPO_COMMIT,
    "code_version": B.VERSION,
    "code_sha256": CODE_SHA256,
    "union_version": U.VERSION,
    "union_sha256": UNION_SHA256,
    "checkpoint_sha256": CHECKPOINT_SHA256,
    "notebook_build": NOTEBOOK_BUILD,
    "lkh_source": LKH_SOURCE,
    "lkh_findtour_sha256": B.LKH_FINDTOUR_SHA256,
    "eax_source": EAX_SOURCE,
    "eax_commit": B.EAX_COMMIT,
    "hardware": HARDWARE,
}
manifest_path = OUTPUT_ROOT / "manifest.json"
if manifest_path.exists():
    saved = json.loads(manifest_path.read_text())
    for key in ["schema", "design", "seeds", "workers", "repo_commit", "code_version", "code_sha256", "union_version",
                "union_sha256", "checkpoint_sha256", "notebook_build", "eax_commit"]:
        assert saved[key] == MANIFEST[key], f"{key} değişti. Ayrı bir EXPERIMENT_NAME kullanın."
    assert saved["protocol"]["sha256"] == PROTOCOL_SHA256, "Protokol değişti. Ayrı bir EXPERIMENT_NAME kullanın."
    assert saved["hardware"]["vcpus_visible"] == HARDWARE["vcpus_visible"], "vCPU sayısı değişti; aynı ortam türünü seçin."
    same_machine_type = (saved["hardware"]["gpu"] == HARDWARE["gpu"]
                         and saved["hardware"]["cpu_model"] == HARDWARE["cpu_model"])
    assert ALLOW_HARDWARE_CHANGE or same_machine_type, \
        (f"Makine türü değişti ({saved['hardware']['gpu']}, {saved['hardware']['cpu_model']} -> {HARDWARE['gpu']}, "
         f"{HARDWARE['cpu_model']}). Aynı türü seçin; bulunamıyorsa ilk hücrede ALLOW_HARDWARE_CHANGE = True yapın "
         "(her örneğin bütün kolları yine tek makinede koşar).")
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

print("Hazır:", CPU_COUNT, "vCPU |", RAM_GB, "GB RAM |", CPU_MODEL, "| GPU:", GPU_NAME, "| LKH:", LKH_SOURCE,
      "| GA-EAX:", EAX_SOURCE, "| SCIP", SCIP_VERSION, "| kod", B.VERSION, CODE_SHA256[:12], U.VERSION,
      UNION_SHA256[:12], "| sonuçlar:", OUTPUT_ROOT)
