"""Build the self-contained, resumable run-v12 notebook (candidate self-correction diagnosis).

    python experiments/v12_self_correction/build_colab_notebook.py

writes notebooks/13_BetaTour_v12_candidate_self_correction.ipynb. The experiment module is embedded byte for byte and
copied to the result archive, so the archive records the exact code that ran.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUTPUT = REPO / "notebooks" / "13_BetaTour_v12_candidate_self_correction.ipynb"

INTRO = """# BetaTour koşu v12 — LKH, POPMUSIC aday listelerinin eksiğini arama sırasında kendisi kapatıyor mu?

**Soru.** BetaTour'un listeleri yakın-optimal tur kenarlarını POPMUSIC'ten çok daha az kaçırıyor, ama LKH-3 bundan hızlanmadı. LKH-3 her yeni en iyi turun kenarlarını aday listelerine kendisi ekliyor (AdjustCandidateSet). Bu defter, bu mekanizmanın POPMUSIC listelerindeki eksiği arama sırasında ne ölçüde kapattığını ölçer.

**Tasarım (protokol v1, Aşama 1).** Referans turu olan 43 örnek (16 × 1.000, 16 × 10.000 düğüm, 11 TSPLIB) ve Tablo 1 bütçeleri. Her örnekte altı yerli, ascent'siz POPMUSIC-5 LKH-3.0.14 süreci (tohum 101–106) paralel koşar; LKH'ye yama yoktur. Her süreç başlangıç aday listesini (C0) yazar; her yeni en iyi tur 20 ms aralıkla kopyalanır. Aynı tohumlarla altı ayrı tek-deneme koşusu, BetaTour ve sayım listelerinin topluluğunu (M = 6) verir. Ölçüler (1.000 düğüm başına): C0'ın, BetaTour ilk-5'in, sayım ilk-5'in ve bütçenin %25/50/75/100'ünde **dinamik aday kümesinin** (C0 + o ana kadarki tüm en iyi tur kenarları) kaçırdığı referans kenarları; C0'da eksik referans kenarların ne kadarını aramanın kendisinin bulduğu; BetaTour listesinde olup LKH'nin hiç ulaşmadığı kenarlar.

**Önceden belirlenen birincil karşılaştırma.** 43 örnek üzerinde, bütçe sonundaki dinamik kümenin ve BetaTour ilk-5 listesinin kaçırdığı referans kenarlar; kesin iki yönlü işaret testi. Dinamik küme daha çok örnekte daha az kaçırır ve p < 0,05 ise H1 (LKH eksiği kendisi kapatır) desteklenir. Sonuç yönünden bağımsız olarak makaleye girer.

**Çalıştırma.** Colab'da **A100 veya L4 GPU** ve **Yüksek RAM** seçin; defter 12'den az vCPU görürse durur. Hücreleri sırayla çalıştırın. Referans turları Drive'daki `MyDrive/TourFold_TSP/eval` kayıtlarından okunur. Tamamlanan örnekler `MyDrive/BetaTour/v12_selfcorrect43` içinde atlanır; kesilen örnek baştan koşar. Sonuç ZIP'i sonunda tarayıcıya iner; denetim ZIP'i Drive'da kalır.

**Tahmini süre.** Kurulum ve LKH derlemesi ≈ 3–5 dk; ısınma ≈ 1 dk; 16 × 5 s + 16 × 120 s + TSPLIB bütçeleri (≈ 35 dk) ≈ **68 dk** çözücü; ilk denemeler, model ve çözümleme ≈ 10–15 dk; arşiv ≈ 1–2 dk. Toplam ≈ **1,5 saat**.
"""

CONFIG = '''EXPERIMENT_NAME = "v12_selfcorrect43"
PROFILE = "paper43"             # 43 instances with reference tours: 16 x 1k, 16 x 10k, 11 TSPLIB
SEEDS = [101, 102, 103, 104, 105, 106]
K_GRAPH = 16
K_LIST = 5
POLL_S = 0.02
FRACTIONS = (0.25, 0.5, 0.75, 1.0)
REQUIRE_VCPUS = 12
REPO_URL = "https://github.com/kurbankotan/betatour-tsp-lkh-candidates.git"
REPO_COMMIT = "d63f7e17942a6af5cba14ad010edca1069d477b3"
CHECKPOINT_SHA256 = "2322f66c752913f6179f8c61f6b50b0f9d21426857d3ec0155e02891fb3ebd70"
LKH_MIRROR_SHA256 = "4c4b9e6a77e50a1eb1eb2c8cb384ed5ef64e0dc88fdbf37bd693798469e1cf33"
EVAL_DIR = "/content/drive/MyDrive/TourFold_TSP/eval"
TSPLIB_NAMES = ["pcb3038", "fl3795", "fnl4461", "rl5915", "rl5934", "pla7397",
                "rl11849", "usa13509", "brd14051", "d15112", "d18512"]
'''

SETUP = '''import base64, hashlib, json, math, os, platform, shutil, subprocess, sys, tarfile, time, urllib.request, uuid, zipfile
from pathlib import Path

TEST_MODE = os.environ.get("BETATOUR_V12_TEST") == "1"     # local tests with a stand-in solver only
if TEST_MODE:
    PROFILE = "smoke"
import importlib.util
for package, module in [("scipy", "scipy"), ("pandas", "pandas")]:
    if importlib.util.find_spec(module) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")
import numpy as np
import pandas as pd
import scipy
import torch

if not TEST_MODE:
    assert torch.cuda.is_available(), "Colab'da GPU runtime seçin: A100 veya L4."
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
CPU_COUNT = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
cpu_max = Path("/sys/fs/cgroup/cpu.max")
if cpu_max.exists():
    quota, period = cpu_max.read_text().split()[:2]
    if quota != "max":
        CPU_COUNT = min(CPU_COUNT, max(1, math.floor(int(quota) / int(period))))
if not TEST_MODE:
    assert CPU_COUNT >= REQUIRE_VCPUS, f"Bu deney {REQUIRE_VCPUS} vCPU ister; görünen: {CPU_COUNT}. A100/L4 + Yüksek RAM seçin."

def cpu_model_name():
    path = Path("/proc/cpuinfo")
    if path.exists():
        for line in path.read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"

WORK_ROOT = Path(os.environ.get("BETATOUR_V12_WORK", "/content/betatour_v12_work"))
WORK_ROOT.mkdir(parents=True, exist_ok=True)
if TEST_MODE:
    REPO = Path(os.environ["BETATOUR_V12_REPO"])
else:
    REPO = WORK_ROOT / "repo"
    if not (REPO / ".git").exists():
        subprocess.run(["git", "clone", "--quiet", REPO_URL, str(REPO)], check=True)
    subprocess.run(["git", "-C", str(REPO), "fetch", "--quiet", "origin", REPO_COMMIT], check=True)
    subprocess.run(["git", "-C", str(REPO), "checkout", "--quiet", REPO_COMMIT], check=True)
    assert subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip() == REPO_COMMIT
sys.path.insert(0, str(REPO / "src"))
import tsplab as T
import tsp_model as M

CHECKPOINT = Path(os.environ.get("BETATOUR_V12_CHECKPOINT", str(REPO / "checkpoints" / "betatour_v2.pt")))
CHECKPOINT_ACTUAL = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
if not TEST_MODE:
    assert CHECKPOINT_ACTUAL == CHECKPOINT_SHA256, "Model dosyası beklenen SHA256 değeriyle eşleşmiyor."
MODEL, CK = M.load_model(str(CHECKPOINT), DEVICE)          # pinned, trusted repository
assert MODEL.cfg.get("use_ensemble") is True

if TEST_MODE:
    OUTPUT_ROOT = Path(os.environ["BETATOUR_V12_OUT"]) / EXPERIMENT_NAME
else:
    from google.colab import drive
    drive.mount("/content/drive")
    OUTPUT_ROOT = Path("/content/drive/MyDrive/BetaTour") / EXPERIMENT_NAME
for sub in ("instances", "audit", "summary", "errors", "code", "archives"):
    (OUTPUT_ROOT / sub).mkdir(parents=True, exist_ok=True)

def get_lkh_3014():
    """Official source first; the pinned copy, checked against its SHA-256, if the official site is unreachable."""
    root = WORK_ROOT / "lkh"
    binary = root / "LKH-3.0.14" / "LKH"
    if binary.exists():
        return str(binary), (root / "source.txt").read_text().strip()
    root.mkdir(parents=True, exist_ok=True)
    options = [("official", "https://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.0.14.tgz", None, 12),
               ("pinned copy", "https://raw.githubusercontent.com/ZhekaiChen/ALCO/main/third_party/LKH3/LKH-3.0.14.tar",
                LKH_MIRROR_SHA256, 90)]
    last = None
    for label, url, expected, timeout in options:
        archive = root / ("LKH-3.0.14.tar" if url.endswith(".tar") else "LKH-3.0.14.tgz")
        try:
            print("LKH kaynağı:", label, url)
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response, archive.open("wb") as f:
                shutil.copyfileobj(response, f)
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            if expected:
                assert digest == expected, f"LKH arşivi SHA256 uyuşmazlığı: {digest}"
            with tarfile.open(archive, "r:*") as tar:
                try:
                    tar.extractall(root, filter="data")
                except TypeError:
                    tar.extractall(root)
            build = subprocess.run(["make", "-j", str(min(4, CPU_COUNT))], cwd=root / "LKH-3.0.14",
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
            if build.returncode != 0 or not binary.exists():
                raise RuntimeError("LKH derlemesi başarısız:\\n" + build.stdout[-3000:])
            (root / "source.txt").write_text(f"{label} {url} sha256={digest}")
            return str(binary), (root / "source.txt").read_text().strip()
        except Exception as exc:
            last = exc
            print(f"{label} başarısız: {exc}")
    raise RuntimeError(f"LKH 3.0.14 indirilemedi/derlenemedi: {last}")

if TEST_MODE:
    LKH, LKH_SOURCE = os.environ["BETATOUR_V12_LKH"], "test stand-in"
else:
    LKH, LKH_SOURCE = get_lkh_3014()
session_path = WORK_ROOT / "runtime_session_id.txt"
if not session_path.exists():
    session_path.write_text(uuid.uuid4().hex)
SESSION_ID = session_path.read_text().strip()
HARDWARE = {"gpu": GPU_NAME, "vcpus_visible": CPU_COUNT, "cpu_model": cpu_model_name(),
            "ram_gb": T.env_info().get("ram_gb"), "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
            "cuda": torch.version.cuda}
print("Hazır:", GPU_NAME, "|", CPU_COUNT, "vCPU | model", sum(p.numel() for p in MODEL.parameters()),
      "parametre | LKH", LKH, "|", LKH_SOURCE, "| sonuçlar", OUTPUT_ROOT)
'''

CODE_CELL = '''BUNDLED_FILES = __BUNDLE__
CODE_ROOT = WORK_ROOT / "code"
for rel, encoded in BUNDLED_FILES.items():
    content = base64.b64decode(encoded)
    for dest in (CODE_ROOT / rel, OUTPUT_ROOT / "code" / rel):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            assert dest.read_bytes() == content, f"Deney kodu değişmiş: {rel}. Yeni EXPERIMENT_NAME kullanın."
        else:
            dest.write_bytes(content)
CODE_SHA256 = {rel: hashlib.sha256(base64.b64decode(encoded)).hexdigest() for rel, encoded in BUNDLED_FILES.items()}
sys.path.insert(0, str(CODE_ROOT / "v12_self_correction"))
import selfcorrect as S
print("Dondurulmuş deney kodu:", S.VERSION, CODE_SHA256)
'''

INSTANCES = '''MANIFEST = {"schema": 1, "experiment_name": EXPERIMENT_NAME, "profile": PROFILE, "seeds": SEEDS,
            "graph_k": K_GRAPH, "candidate_k": K_LIST, "poll_s": POLL_S, "fractions": list(FRACTIONS),
            "repo_commit": REPO_COMMIT, "checkpoint_sha256": CHECKPOINT_ACTUAL, "lkh_source": LKH_SOURCE,
            "lkh_binary_sha256": S.sha256_file(LKH), "code_sha256": CODE_SHA256, "module_version": S.VERSION,
            "hardware": HARDWARE,
            "primary": "per instance, reference edges missing from the dynamic candidate set at the budget versus "
                       "from the BetaTour top-5 list (per 1,000 nodes); exact two-sided sign test; H1 supported if "
                       "the dynamic set misses fewer on more instances and p < 0.05",
            "lkh_params": {"CANDIDATE_SET_TYPE": "POPMUSIC", "MAX_CANDIDATES": K_LIST, "SUBGRADIENT": "NO",
                           "RUNS": 1, "MAX_TRIALS": 100000000, "TRACE_LEVEL": 1, "STOP_AT_OPTIMUM": "NO",
                           "TIME_LIMIT": "10 x budget (external monotonic deadline at the budget)"}}
manifest_path = OUTPUT_ROOT / "manifest.json"
if manifest_path.exists():
    saved = json.loads(manifest_path.read_text())
    for key in ["schema", "profile", "seeds", "graph_k", "candidate_k", "poll_s", "fractions", "repo_commit",
                "checkpoint_sha256", "code_sha256", "module_version", "lkh_params", "primary"]:
        assert saved[key] == MANIFEST[key], f"{key} değişti. Ayrı EXPERIMENT_NAME kullanın."
    for key in ["gpu", "vcpus_visible"]:
        assert saved["hardware"][key] == HARDWARE[key], f"Donanım {key} değişti; aynı Colab türünü seçin."
else:
    S.atomic_json(manifest_path, MANIFEST)

PROBLEM_DIR = WORK_ROOT / "problems"
PROBLEM_DIR.mkdir(exist_ok=True)

def make_instances():
    out = []
    if PROFILE == "smoke":
        rng = np.random.default_rng(2024)
        for name, dist in (("uni1k_000", "uniform"), ("clu1k_000", "clustered")):
            xy = T.to_int_coords(T.gen_instance(180, dist, rng))
            out.append(dict(group=name[:5], name=name, xy=xy, ew="EUC_2D", tsplib=None, budget=2.0))
        path = Path(os.environ["BETATOUR_V12_TSPLIB"]) / "ceil160.tsp"
        data = T.read_tsplib(str(path))
        out.append(dict(group="tsplib", name="ceil160", xy=data["xy"], ew=data["edge_weight_type"],
                        tsplib=str(path), budget=2.0))
    else:
        specs = [("uni1k", 1000, 8, "uniform", 101, 5.0), ("clu1k", 1000, 8, "clustered", 202, 5.0),
                 ("uni10k", 10000, 8, "uniform", 303, 120.0), ("clu10k", 10000, 8, "clustered", 404, 120.0)]
        for group, n, count, dist, gen_seed, budget in specs:
            rng = np.random.default_rng(gen_seed)
            for i in range(count):
                xy = T.to_int_coords(T.gen_instance(n, dist, rng))
                out.append(dict(group=group, name=f"{group}_{i:03d}", xy=xy, ew="EUC_2D", tsplib=None,
                                budget=budget))
        paths = T.download_tsplib(str(WORK_ROOT / "tsplib"), names=TSPLIB_NAMES)
        absent = sorted(set(TSPLIB_NAMES) - set(paths))
        assert not absent, f"Eksik TSPLIB dosyaları: {absent}. Ağ bağlantısını kontrol edip yeniden çalıştırın."
        for name in TSPLIB_NAMES:
            data = T.read_tsplib(paths[name])
            n = int(data["dimension"])
            assert len(data["xy"]) == n and data["edge_weight_type"] in {"EUC_2D", "CEIL_2D"}, name
            out.append(dict(group="tsplib", name=name, xy=data["xy"], ew=data["edge_weight_type"],
                            tsplib=str(paths[name]), budget=max(10.0, 0.02 * n)))
    for inst in out:
        if inst["tsplib"]:
            inst["problem"] = Path(inst["tsplib"])
        else:
            inst["problem"] = PROBLEM_DIR / f"{inst['name']}.tsp"
            if not inst["problem"].exists():
                T.write_tsp(str(inst["problem"]), inst["xy"], name=inst["name"], ew="EUC_2D")
        inst["xy_sha256"] = S.xy_sha256(inst["xy"])
    out.sort(key=lambda x: (x["budget"], x["name"]))
    return out

def load_references(eval_dir, wanted):
    refs = {}
    for shard_path in sorted(Path(eval_dir).glob("*_part*.npz")):
        shard = T.Shard(str(shard_path))
        for i in range(len(shard)):
            rec = shard[i]
            name = str(rec.get("name", ""))
            if name in wanted and name not in refs and "tour" in rec and T.is_valid_tour(rec["tour"], rec["n"]):
                refs[name] = (np.asarray(rec["tour"], dtype=np.int64), np.asarray(rec["xy"]), shard_path.name)
    return refs

ALL = make_instances()
REFS = load_references(os.environ.get("BETATOUR_V12_EVAL", EVAL_DIR), {x["name"] for x in ALL})
INSTANCES, EXCLUDED = [], []
for inst in ALL:
    ref = REFS.get(inst["name"])
    if ref is None:
        EXCLUDED.append((inst["name"], "no reference tour"))
        continue
    tour, ref_xy, source = ref
    same = (np.array_equal(np.asarray(ref_xy, dtype=np.int64), inst["xy"]) if inst["tsplib"] is None
            else np.allclose(np.asarray(ref_xy, dtype=np.float64), inst["xy"]))
    if not same:
        EXCLUDED.append((inst["name"], "reference coordinates differ"))
        continue
    inst["ref_tour"], inst["reference_source"] = tour, f"eval shard {source}"
    INSTANCES.append(inst)
print(f"{len(INSTANCES)} örnek referans turuyla hazır; dışarıda kalan: {EXCLUDED or 'yok'}")
if PROFILE == "paper43" and len(INSTANCES) != 43:
    print("UYARI: protokol 43 örnek bekliyor; eksikler sonuç raporunda listelenecek.")
budget_total = sum(x["budget"] for x in INSTANCES)
print(f"Çözücü süresi ≈ {budget_total / 60:.0f} dk; ilk denemeler, model ve çözümleme ile toplam ≈ "
      f"{1.2 * budget_total / 60 + 5:.0f} dk.")
'''

WARMUP = '''# Warm-up on a small instance: LKH writes C0 and rewrites its best tour, the first-trial run reproduces the
# native run's C0 (same seed), the model produces sensible lists, and the analysis runs end to end.
warm_dir = WORK_ROOT / "warmup"
shutil.rmtree(warm_dir, ignore_errors=True)
warm_dir.mkdir(parents=True)
warm_xy = T.to_int_coords(T.gen_instance(200, "uniform", np.random.default_rng(9090)))
warm_problem = warm_dir / "warm.tsp"
T.write_tsp(str(warm_problem), warm_xy, name="warm", ew="EUC_2D")
t0 = time.perf_counter()
warm_first = S.run_first_trials(T, LKH, warm_problem, warm_xy, "EUC_2D", warm_dir / "first", SEEDS, k=K_LIST)
warm_lists = S.model_and_count_lists(M, T, MODEL, DEVICE, warm_xy, warm_first["tours"], warm_first["costs"],
                                     K_GRAPH, K_LIST)
warm_native = S.run_native_pop(T, LKH, warm_problem, warm_xy, "EUC_2D", warm_dir / "native", SEEDS, 3.0,
                               k=K_LIST, poll_s=POLL_S)
for m, seed in enumerate(SEEDS):
    raw = warm_native["seeds"][seed]
    assert np.array_equal(raw["c0"], warm_first["cands"][m]), f"tohum {seed}: C0 ilk-deneme koşusuyla aynı değil"
    caps, _ = S.decode_captures(T, warm_xy, "EUC_2D", raw["captures"], 3.0)
    assert caps, f"tohum {seed}: hiç tur yakalanmadı"
best_first = warm_first["tours"][int(np.argmin(warm_first["costs"]))]
cover = 1.0 - T.candidate_coverage(warm_lists["beta"], best_first, ks=[K_LIST])[K_LIST]["missing"] / len(warm_xy)
assert cover >= 0.90 or TEST_MODE, f"Model listeleri beklenmedik: en iyi ilk turun kenarlarının yalnızca %{100 * cover:.1f}'i ilk-5'te"
ref_warm = min((S.decode_captures(T, warm_xy, "EUC_2D", warm_native["seeds"][s]["captures"], 3.0)[0][-1]
                for s in SEEDS), key=lambda r: r[1])[2]
warm_summary, _ = S.analyse_instance(T, dict(name="warm", group="warm", xy=warm_xy, ew="EUC_2D"), ref_warm,
                                     warm_first, warm_native, warm_lists, 3.0, FRACTIONS)
print(f"Isınma tamam ({time.perf_counter() - t0:.1f} s): yakalanan tur sayıları",
      [r["captures"] for r in warm_summary["seed_rows"]], "| BetaTour ilk-5, en iyi ilk turun kenarlarının "
      f"%{100 * cover:.1f}'ini içeriyor")
shutil.rmtree(warm_dir, ignore_errors=True)
'''

RUN = '''def finished(name):
    path = OUTPUT_ROOT / "instances" / f"{name}.json"
    if not path.exists():
        return False
    rec = json.loads(path.read_text())
    return rec.get("complete") is True and rec.get("code_sha256") == CODE_SHA256

def save_progress(state):
    done = [x["name"] for x in INSTANCES if finished(x["name"])]
    S.atomic_json(OUTPUT_ROOT / "progress.json", {"state": state, "completed": len(done), "total": len(INSTANCES),
                                                  "excluded": EXCLUDED, "session": SESSION_ID,
                                                  "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})

save_progress("running")
for ordinal, inst in enumerate(INSTANCES, 1):
    name = inst["name"]
    if finished(name):
        print(f"[{ordinal}/{len(INSTANCES)}] {name}: tamamlanmış, atlandı")
        continue
    work = WORK_ROOT / "runs" / name
    shutil.rmtree(work, ignore_errors=True)
    t0 = time.perf_counter()
    try:
        first = S.run_first_trials(T, LKH, inst["problem"], inst["xy"], inst["ew"], work / "first", SEEDS, k=K_LIST)
        lists = S.model_and_count_lists(M, T, MODEL, DEVICE, inst["xy"], first["tours"], first["costs"],
                                        K_GRAPH, K_LIST)
        native = S.run_native_pop(T, LKH, inst["problem"], inst["xy"], inst["ew"], work / "native", SEEDS,
                                  inst["budget"], k=K_LIST, poll_s=POLL_S)
        summary, audit = S.analyse_instance(T, inst, inst["ref_tour"], first, native, lists, inst["budget"],
                                            FRACTIONS)
        if len(inst["xy"]) >= 1000 and summary["per1000"]["missing_beta"] > 20 and not TEST_MODE:
            raise AssertionError("BetaTour listeleri beklenmedik derecede çok referans kenarı kaçırıyor; model yüklemesini kontrol edin")
        summary.update(complete=True, code_sha256=CODE_SHA256, module_version=S.VERSION, xy_sha256=inst["xy_sha256"],
                       session=SESSION_ID, hardware=HARDWARE, wall_s=time.perf_counter() - t0,
                       finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        S.atomic_npz(OUTPUT_ROOT / "audit" / f"{name}.npz", **audit)
        S.atomic_json(OUTPUT_ROOT / "instances" / f"{name}.json", summary)
    except Exception as exc:
        S.atomic_json(OUTPUT_ROOT / "errors" / f"{name}.json",
                      {"name": name, "error": repr(exc), "session": SESSION_ID,
                       "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        save_progress("error")
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)
    p = summary["per1000"]
    print(f"[{ordinal}/{len(INSTANCES)}] {name}: C0 {p['missing_c0']:.2f} | BetaTour {p['missing_beta']:.2f} | "
          f"sayım {p['missing_count']:.2f} | dinamik %25 {p['missing_dynamic_25']:.2f} → %100 "
          f"{p['missing_dynamic_100']:.2f} (1.000 düğüm başına kaçırılan referans kenarı) | {summary['wall_s']:.0f} s")
save_progress("complete" if all(finished(x["name"]) for x in INSTANCES) else "partial")
'''

ANALYSIS = '''RECORDS = []
for inst in INSTANCES:
    path = OUTPUT_ROOT / "instances" / f"{inst['name']}.json"
    if path.exists():
        rec = json.loads(path.read_text())
        if rec.get("complete") and rec.get("code_sha256") == CODE_SHA256:
            RECORDS.append(rec)
assert RECORDS, "Tamamlanmış örnek yok."
INST_TABLE, GROUP_TABLE, PRIMARY = S.summary_tables(RECORDS)
PRIMARY["complete_instances"] = len(RECORDS)
PRIMARY["expected_instances"] = len(INSTANCES)
INST_TABLE.to_csv(OUTPUT_ROOT / "summary" / "instance_table.csv", index=False)
GROUP_TABLE.to_csv(OUTPUT_ROOT / "summary" / "group_table.csv")
S.atomic_json(OUTPUT_ROOT / "summary" / "primary.json", PRIMARY)
flag_rows = [{"name": r["name"], "seed": s, "flags": ";".join(f)} for r in RECORDS for s, f in r["flags"].items()]
pd.DataFrame(flag_rows, columns=["name", "seed", "flags"]).to_csv(OUTPUT_ROOT / "summary" / "flags.csv", index=False)
pd.set_option("display.width", 220)
cols = ["missing_c0", "missing_beta", "missing_count", "missing_dynamic_25", "missing_dynamic_50",
        "missing_dynamic_100", "recovered_share_100", "r0_in_final_share", "rc_in_final_share",
        "r0_beta_not_reached", "final_gap_to_reference_pct"]
print("1.000 düğüm başına kaçırılan referans kenarı ve oranlar (örnek ortalamaları):")
print(GROUP_TABLE[cols].round(3).to_string())
print()
print("Birincil karşılaştırma:", json.dumps(PRIMARY, indent=2, ensure_ascii=False))
if len(RECORDS) < len(INSTANCES):
    print(f"ARA SONUÇ: {len(RECORDS)}/{len(INSTANCES)} örnek tamamlandı.")
'''

ARCHIVE = '''tag = EXPERIMENT_NAME
result_zip = OUTPUT_ROOT / "archives" / f"{tag}_results.zip"
audit_zip = OUTPUT_ROOT / "archives" / f"{tag}_audit.zip"
def add_tree(z, root, rel_root=None, pattern="*"):
    for path in sorted(Path(root).rglob(pattern)):
        if path.is_file() and ".tmp" not in path.name:
            z.write(path, str(path.relative_to(rel_root or OUTPUT_ROOT)))
for target, parts in ((result_zip, ["manifest.json", "progress.json", "instances", "summary", "code", "errors"]),
                      (audit_zip, ["manifest.json", "audit"])):
    tmp = target.with_name(target.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for part in parts:
            p = OUTPUT_ROOT / part
            if p.is_file():
                z.write(p, part)
            elif p.is_dir():
                add_tree(z, p)
    os.replace(tmp, target)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
    print(f"{target.name}: {target.stat().st_size / 1e6:.1f} MB, SHA-256 {S.sha256_file(target)}")
local_copy = WORK_ROOT / result_zip.name
shutil.copy(result_zip, local_copy)
if not TEST_MODE:
    from google.colab import files
    files.download(str(local_copy))
print("Sonuç ZIP'i indirildi; denetim ZIP'i Drive'da:", audit_zip)
'''


def code_cell(source: str, index: int) -> dict:
    compile(source.replace("__BUNDLE__", "{}"), f"<v12 cell {index}>", "exec")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "id": f"v12_{index:02d}_{hashlib.sha256(source.encode()).hexdigest()[:10]}",
            "source": source.splitlines(keepends=True)}


def make_notebook() -> dict:
    files = {"v12_self_correction/selfcorrect.py": HERE / "selfcorrect.py",
             "v12_self_correction/build_colab_notebook.py": HERE / "build_colab_notebook.py"}
    bundle = {rel: base64.b64encode(path.read_bytes()).decode("ascii") for rel, path in files.items()}
    cells = [{"cell_type": "markdown", "metadata": {}, "id": "v12_00_intro", "source": INTRO.splitlines(keepends=True)}]
    for i, src in enumerate([CONFIG, SETUP, CODE_CELL.replace("__BUNDLE__", repr(bundle)), INSTANCES, WARMUP, RUN,
                             ANALYSIS, ARCHIVE], 1):
        cells.append(code_cell(src, i))
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm",
                                                                         "provenance": []},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(make_notebook(), indent=1, ensure_ascii=False) + "\n")
    print("written", OUTPUT)
