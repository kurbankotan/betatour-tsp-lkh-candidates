"""Generate the self-contained BetaTour v9 Colab pilot notebook.

The notebook embeds our small policy/coordinator/patch files. It downloads
the unmodified LKH source and the frozen BetaTour checkpoint at runtime.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
REPO = EXPERIMENTS.parent
OUTPUT = REPO.parents[1] / "outputs" / "BetaTour_v9_CanliLKH_Pilot_Colab.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    compile(source, "<notebook cell>", "exec")
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": source.splitlines(keepends=True)}


FILES = {
    "livehook/BetaTourHook.h": EXPERIMENTS / "livehook" / "BetaTourHook.h",
    "livehook/build_livehook.py": EXPERIMENTS / "livehook" / "build_livehook.py",
    "livehook/coordinator.py": EXPERIMENTS / "livehook" / "coordinator.py",
    "policy/adaptive_policy.py": EXPERIMENTS / "policy" / "adaptive_policy.py",
    "v9_live/live_pilot.py": HERE / "live_pilot.py",
}


def make_notebook() -> dict:
    bundle = {name: base64.b64encode(path.read_bytes()).decode("ascii")
              for name, path in FILES.items()}
    cells = [
        markdown("""# BetaTour v9 — canlı LKH müdahalesi, pilot deney

Bu pilot, **eski v8 örneklerinin sekizini** kullanır; sonuçları yeni bir doğrulama kümesi olarak sunmayacağız. Amaç, LKH sürecini yeniden başlatmadan TourFold tahminini aramaya eklemek ve bunun gerçekten işe yarayıp yaramadığını görmek. Altı LKH tohumu ilk turlarını ürettikten sonra aynı süreçler devam eder. Her kolun 120/300 saniyelik **duvar saati** sınırına model çıkarımı ve bekleme dahildir.

Çalıştırma: Colab'da **A100 veya L4 GPU**, High RAM ve en az **12 görünür vCPU** seçin. Pilota başladıktan sonra aynı GPU türünü ve vCPU sayısını kullanın. CPU modeli değişirse her kolun donanımı ayrı kaydedilir ve karışık örnekler analizde işaretlenir. Hücreleri sırayla çalıştırın. Her tamamlanan kol `MyDrive/BetaTour/v9_live_pilot` içine kaydolur. Oturum kesilirse yeniden çalıştırın; tamamlanan kollar atlanır, yarım kalan kol tekrarlanır. Bütün pilot bitince sonuç ZIP'i tarayıcıdan otomatik indirilir.

Kollar: kesintisiz POPMUSIC-5/6, canlı müdahalesiz kontrol, BetaTour ile mevcut adayın öne alınması, seçili düğümlere seyrek BetaTour kenarı ve aynı düğümlerde tur sıklığına dayalı kenar. Pilot politika seçimi içindir; sonraki bağımsız test ayrı bir çalışma olacaktır.
"""),
        code("""EXPERIMENT_NAME = "v9_live_pilot"
SEEDS = list(range(301, 307))
ARMS = ["native_pop5", "native_pop6", "sham", "beta_promote", "beta_sparse", "frequency_same_gate"]
INSTANCE_SPECS = [
    ["uni10k", 10000, 2, "uniform", 4107, 120.0],
    ["clu10k", 10000, 2, "clustered", 4208, 120.0],
    ["uni20k", 20000, 2, "uniform", 4309, 300.0],
    ["clu20k", 20000, 2, "clustered", 4410, 300.0],
]
EXPECTED_XY_SHA256 = {
    "augment_uni10k_000": "0d5d680b465faa68d33467c4f70bb5ad1e52b25a2d444066ab922130458efcd6",
    "augment_uni10k_001": "00747175507524f481bd6e34ea55f0a5474d3dcd642a67543d871d9a64b53796",
    "augment_clu10k_000": "072fc96ec668bb9c95b3a52562f23cedb0b2e89494946d159243b638b7b3dc8e",
    "augment_clu10k_001": "965fe4315060118d313a357404f750a23ea89cc11c174cb1138a9f32b312ca22",
    "augment_uni20k_000": "d0dbad4fdd4151029d3bf59cca81c19248129c87030003d7c455e37e32b7d674",
    "augment_uni20k_001": "c35e5c135b527d91c5c9b6258320f35526c2e735a50d581c48e96150e9cb3173",
    "augment_clu20k_000": "abdd6474a4cd0e25ee394c227ef8f5c4beb5c09c8a24abddcad273d0dc39181e",
    "augment_clu20k_001": "ee3edf42dbbb94cd344d1e3bef06d5674e47330e937a8acda4fd069f4faf79ff",
}
MAX_NEW_ARMS = None  # Örn. 3: bu oturumda en fazla üç yeni kol; sonra tekrar çalıştırın.
SPARSE_FRACTION = 0.25
REPO_COMMIT = "d63f7e17942a6af5cba14ad010edca1069d477b3"
CHECKPOINT_SHA256 = "2322f66c752913f6179f8c61f6b50b0f9d21426857d3ec0155e02891fb3ebd70"
LKH_TAR_SHA256 = "4c4b9e6a77e50a1eb1eb2c8cb384ed5ef64e0dc88fdbf37bd693798469e1cf33"
LKH_FINDTOUR_SHA256 = "567e72acc9b7392488dd158be4df0267560424204a0e88d2d9b62f923a7c0eee"
assert len(SEEDS) == 6 and len(EXPECTED_XY_SHA256) == 8
"""),
        code("""import base64, hashlib, importlib.util, json, math, os, platform, shutil, subprocess, sys, tarfile, time, urllib.request, uuid, zipfile
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
for package in ("scipy", "pandas"):
    if importlib.util.find_spec(package) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)
import numpy as np
import scipy
from scipy.stats import binomtest
import pandas as pd
import torch
from IPython.display import display

assert torch.cuda.is_available(), "Colab GPU runtime seçin (A100 veya L4)."
torch.set_num_threads(1)
DEVICE = torch.device("cuda")
GPU_NAME = torch.cuda.get_device_name(0)
assert "A100" in GPU_NAME or "L4" in GPU_NAME, f"Beklenen GPU A100/L4; görünen: {GPU_NAME}"
CPU_COUNT = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
cpu_max = Path("/sys/fs/cgroup/cpu.max")
if cpu_max.exists():
    quota, period = cpu_max.read_text().split()[:2]
    if quota != "max":
        CPU_COUNT = min(CPU_COUNT, max(1, math.floor(int(quota) / int(period))))
assert CPU_COUNT >= 12, f"En az 12 görünür vCPU gerekli; görünen: {CPU_COUNT}"
def cpu_model_name():
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"

WORK_ROOT = Path("/content/betatour_v9_live_work")
WORK_ROOT.mkdir(parents=True, exist_ok=True)
REPO = WORK_ROOT / "repo"
if not (REPO / ".git").exists():
    subprocess.run(["git", "clone", "--quiet", "https://github.com/kurbankotan/betatour-tsp-lkh-candidates.git", str(REPO)], check=True)
subprocess.run(["git", "-C", str(REPO), "fetch", "--quiet", "origin", REPO_COMMIT], check=True)
subprocess.run(["git", "-C", str(REPO), "checkout", "--quiet", REPO_COMMIT], check=True)
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
HARDWARE = {"gpu": GPU_NAME, "vcpus": CPU_COUNT, "cpu_model": cpu_model_name(),
            "python": platform.python_version(), "torch": torch.__version__}
print("Hazır:", HARDWARE, "Drive:", OUTPUT_ROOT)
"""),
        code("""BUNDLED_FILES = """ + repr(bundle) + """
CODE_ROOT = WORK_ROOT / "code"
for rel, encoded in BUNDLED_FILES.items():
    dest = CODE_ROOT / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    content = base64.b64decode(encoded)
    if dest.exists():
        assert dest.read_bytes() == content, f"Kod dosyası değişti: {rel}"
    else:
        dest.write_bytes(content)
    drive_dest = OUTPUT_ROOT / "code" / rel
    drive_dest.parent.mkdir(parents=True, exist_ok=True)
    if drive_dest.exists():
        assert drive_dest.read_bytes() == content, f"Drive kod dosyası değişti: {rel}"
    else:
        drive_dest.write_bytes(content)
CODE_SHA256 = {rel: hashlib.sha256(base64.b64decode(encoded)).hexdigest()
               for rel, encoded in BUNDLED_FILES.items()}
sys.path[:0] = [str(CODE_ROOT / "livehook"), str(CODE_ROOT / "policy"), str(CODE_ROOT / "v9_live")]
import build_livehook
from coordinator import read_tour
from live_pilot import run_live_arm, run_native_arm, sha256_file
print("Deney kodu doğrulandı:", CODE_SHA256)
"""),
        code("""LKH_ROOT = WORK_ROOT / "lkh"
LKH_ROOT.mkdir(exist_ok=True)
SOURCE_DIR = LKH_ROOT / "LKH-3.0.14"
SOURCE_INFO_PATH = LKH_ROOT / "source_info.json"
if not SOURCE_DIR.exists():
    sources = [
        ("official", "https://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.0.14.tgz", None, 12),
        ("pinned_mirror", "https://raw.githubusercontent.com/ZhekaiChen/ALCO/main/third_party/LKH3/LKH-3.0.14.tar", LKH_TAR_SHA256, 90),
    ]
    last_error = None
    for label, url, expected_sha, timeout_s in sources:
        archive = LKH_ROOT / ("LKH-3.0.14.tgz" if url.endswith(".tgz") else "LKH-3.0.14.tar")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=timeout_s) as response, archive.open("wb") as out:
                shutil.copyfileobj(response, out)
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            if expected_sha:
                assert digest == expected_sha, f"LKH arşiv SHA256 uyuşmuyor: {digest}"
            with tarfile.open(archive, "r:*") as tar:
                try:
                    tar.extractall(LKH_ROOT, filter="data")
                except TypeError:
                    tar.extractall(LKH_ROOT)
            assert SOURCE_DIR.exists()
            actual_source_sha = hashlib.sha256((SOURCE_DIR / "SRC/FindTour.c").read_bytes()).hexdigest()
            assert actual_source_sha == LKH_FINDTOUR_SHA256, f"LKH 3.0.14 kaynağı beklenenden farklı: {actual_source_sha}"
            SOURCE_INFO_PATH.write_text(json.dumps({"label": label, "url": url, "archive_sha256": digest}, indent=2))
            break
        except Exception as exc:
            last_error = exc
            if SOURCE_DIR.exists():
                shutil.rmtree(SOURCE_DIR)
    else:
        raise RuntimeError(f"LKH indirilemedi: {last_error}")
assert hashlib.sha256((SOURCE_DIR / "SRC/FindTour.c").read_bytes()).hexdigest() == LKH_FINDTOUR_SHA256
if not SOURCE_INFO_PATH.exists():
    cached = next((p for p in (LKH_ROOT / "LKH-3.0.14.tar", LKH_ROOT / "LKH-3.0.14.tgz") if p.exists()), None)
    assert cached is not None, "LKH kaynak kayıt dosyası eksik; kaynak arşiv bulunamadı"
    SOURCE_INFO_PATH.write_text(json.dumps({"label": "recovered_cache", "url": "local_cache",
                                            "archive_sha256": hashlib.sha256(cached.read_bytes()).hexdigest()}, indent=2))
NATIVE_LKH = SOURCE_DIR / "LKH"
if not NATIVE_LKH.exists():
    with (LKH_ROOT / "native_build.log").open("w") as build_log:
        subprocess.run(["make", "-j", str(min(6, CPU_COUNT))], cwd=SOURCE_DIR, check=True,
                       stdout=build_log, stderr=subprocess.STDOUT)
PATCHED_DIR = LKH_ROOT / "LKH-3.0.14-BetaTour"
PATCHED_LKH = PATCHED_DIR / "LKH"
if not PATCHED_LKH.exists():
    if PATCHED_DIR.exists():
        shutil.rmtree(PATCHED_DIR)
    build_livehook.build(SOURCE_DIR, PATCHED_DIR, jobs=min(6, CPU_COUNT))
BUILD_INFO = {"native_sha256": sha256_file(NATIVE_LKH),
              "patched_sha256": sha256_file(PATCHED_LKH),
              "upstream_findtour_sha256": LKH_FINDTOUR_SHA256}
origin_path = OUTPUT_ROOT / "code" / "lkh_source_info.json"
if not origin_path.exists():
    origin_path.write_text(SOURCE_INFO_PATH.read_text())
print("LKH 3.0.14 derlendi:", BUILD_INFO)
"""),
        code("""def write_json_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
    os.replace(tmp, path)

MANIFEST = {"schema": 1, "profile": "pilot_old_v8_instances",
            "experiment_name": EXPERIMENT_NAME, "arms": ARMS, "seeds": SEEDS,
            "instance_specs": INSTANCE_SPECS, "xy_hashes": EXPECTED_XY_SHA256,
            "sparse_fraction": SPARSE_FRACTION, "repo_commit": REPO_COMMIT,
            "checkpoint_sha256": CHECKPOINT_SHA256, "hardware": HARDWARE,
            "code_sha256": CODE_SHA256, "build": BUILD_INFO,
            "analysis_plan": "Pilot only; compare endpoints descriptively and choose one frozen policy before a new holdout."}
manifest_path = OUTPUT_ROOT / "manifest.json"
if manifest_path.exists():
    saved = json.loads(manifest_path.read_text())
    for key in MANIFEST:
        if key != "hardware":
            assert saved[key] == MANIFEST[key], f"Drive'daki pilotun {key} ayarı değişti. Yeni EXPERIMENT_NAME kullanın."
    for key in ("gpu", "vcpus"):
        assert saved["hardware"][key] == HARDWARE[key], f"Donanım {key} değişti; eşli pilot için aynı türü seçin."
    if saved["hardware"] != HARDWARE:
        print("UYARI: CPU modeli veya kütüphane sürümü değişti; kol donanımları ayrı kaydedilecek.")
else:
    write_json_atomic(manifest_path, MANIFEST)

# This warms CUDA kernels and verifies the first six tours against native LKH.
warm_xy = T.to_int_coords(T.gen_instance(100, "uniform", np.random.default_rng(9090)))
warm_succ = np.stack([np.roll(np.arange(100), -1)] * 6, axis=1).astype(np.int32)
M.predict_tourfold(MODEL, warm_xy, 16, DEVICE, ens_succ=warm_succ, ens_cost=np.ones(6))
warm_problem = WORK_ROOT / "warmup.tsp"
T.write_tsp(warm_problem, warm_xy)
warm_dir = WORK_ROOT / "warmup" / uuid.uuid4().hex
warm_dir.parent.mkdir(exist_ok=True)
warm = run_live_arm("sham", PATCHED_LKH, warm_problem, warm_xy, warm_dir,
                    SEEDS, 15.0, MODEL, DEVICE, T, M, max_trials=2)
def edge_set(tour):
    tour = np.asarray(tour, dtype=np.int32)
    return set(zip(np.minimum(tour, np.roll(tour, -1)),
                   np.maximum(tour, np.roll(tour, -1))))
for seed in SEEDS:
    native = T.run_lkh(str(NATIVE_LKH), str(warm_problem),
                       {"CANDIDATE_SET_TYPE": "POPMUSIC", "MAX_CANDIDATES": 5,
                        "SUBGRADIENT": "NO", "SEED": seed, "MAX_TRIALS": 1,
                        "TIME_LIMIT": 30, "STOP_AT_OPTIMUM": "NO"},
                       str(WORK_ROOT / "warmup" / f"native_{seed}_{uuid.uuid4().hex}"), timeout=30)
    patch_first = np.asarray(read_tour(warm_dir / f"seed_{seed}" / "first.tour")) - 1
    assert edge_set(native["tour"]) == edge_set(patch_first), f"İlk tur uyuşmuyor: {seed}"
print("GPU, yama, altı süreç ve ilk tur eşitliği doğrulandı.")
"""),
        code("""INSTANCES = []
for group, n, count, distribution, generator_seed, budget_s in INSTANCE_SPECS:
    rng = np.random.default_rng(generator_seed)
    for index in range(count):
        xy = T.to_int_coords(T.gen_instance(n, distribution, rng))
        name = f"augment_{group}_{index:03d}"
        digest = hashlib.sha256(np.ascontiguousarray(xy).tobytes()).hexdigest()
        assert digest == EXPECTED_XY_SHA256[name], f"Eski v8 koordinatları uyuşmuyor: {name}"
        INSTANCES.append({"name": name, "group": group, "n": n,
                          "budget_s": budget_s, "xy": xy, "xy_sha256": digest})
INSTANCES.sort(key=lambda item: (item["budget_s"], item["name"]))
assert len(INSTANCES) == 8
def arm_order(name):
    return sorted(ARMS, key=lambda arm: hashlib.sha256((name + ":" + arm).encode()).hexdigest())
print("Pilot:", len(INSTANCES), "örnek,", len(ARMS), "kol; kesintisiz tek kol bütçesi",
      sum(item["budget_s"] for item in INSTANCES) / 3600, "saat")
"""),
        code("""def arm_record_path(inst, arm):
    return OUTPUT_ROOT / "instances" / inst["name"] / arm / "result.json"

def validate_saved(inst, arm, record):
    assert record["complete"] and record["arm"] == arm and record["instance"] == inst["name"]
    assert record["xy_sha256"] == inst["xy_sha256"] and record["seeds"] == SEEDS
    assert record["budget_s"] == inst["budget_s"] and record["build"] == BUILD_INFO
    assert len(record["seed_costs"]) == 6 and min(record["seed_costs"]) == record["best_cost"]
    return record

def save_progress(state="running", last=None):
    completed = sum(arm_record_path(inst, arm).exists() for inst in INSTANCES for arm in ARMS)
    write_json_atomic(OUTPUT_ROOT / "progress.json", {"state": state, "completed_arms": completed,
                      "total_arms": len(INSTANCES) * len(ARMS), "last": last,
                      "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})

new_arms = 0
save_progress()
for inst in INSTANCES:
    instance_dir = OUTPUT_ROOT / "instances" / inst["name"]
    instance_dir.mkdir(parents=True, exist_ok=True)
    coord_path = instance_dir / "xy.npz"
    if not coord_path.exists():
        temp = instance_dir / "xy.tmp.npz"
        np.savez_compressed(temp, xy=inst["xy"])
        os.replace(temp, coord_path)
    with np.load(coord_path, allow_pickle=False) as saved_xy:
        assert hashlib.sha256(np.ascontiguousarray(saved_xy["xy"]).tobytes()).hexdigest() == inst["xy_sha256"]
    local_problem = WORK_ROOT / "problems" / (inst["name"] + ".tsp")
    local_problem.parent.mkdir(exist_ok=True)
    T.write_tsp(local_problem, inst["xy"])
    for arm in arm_order(inst["name"]):
        record_path = arm_record_path(inst, arm)
        if record_path.exists():
            validate_saved(inst, arm, json.loads(record_path.read_text()))
            print("ATLA:", inst["name"], arm, flush=True)
            continue
        if MAX_NEW_ARMS is not None and new_arms >= MAX_NEW_ARMS:
            break
        attempt_id = uuid.uuid4().hex
        local_run = WORK_ROOT / "runs" / inst["name"] / arm / attempt_id
        local_run.parent.mkdir(parents=True, exist_ok=True)
        try:
            print("BAŞLA:", inst["name"], arm, "bütçe", inst["budget_s"], "sn", flush=True)
            if arm.startswith("native_"):
                raw = run_native_arm(arm, NATIVE_LKH, local_problem, inst["xy"],
                                     local_run, SEEDS, inst["budget_s"], T)
            else:
                raw = run_live_arm(arm, PATCHED_LKH, local_problem, inst["xy"],
                                   local_run, SEEDS, inst["budget_s"], MODEL, DEVICE,
                                   T, M, sparse_fraction=SPARSE_FRACTION)
            drive_arm = record_path.parent
            drive_arm.mkdir(parents=True, exist_ok=True)
            audit_rel = f"instances/{inst['name']}/{arm}/{attempt_id}"
            audit_dest = OUTPUT_ROOT / audit_rel
            shutil.copytree(local_run, audit_dest)
            local_policy = local_run.parent / (local_run.name + "_audit")
            if local_policy.exists():
                shutil.copytree(local_policy, audit_dest / "policy_audit")
            record = {"schema": 1, "complete": True, "instance": inst["name"],
                      "group": inst["group"], "n": inst["n"], "xy_sha256": inst["xy_sha256"],
                      "arm": arm, "seeds": SEEDS, "budget_s": inst["budget_s"],
                      "seed_costs": raw["seed_costs"], "best_cost": raw["best_cost"],
                      "wall_elapsed_s": raw["wall_elapsed_s"], "build": BUILD_INFO,
                      "hardware": HARDWARE,
                      "audit_relpath": audit_rel, "attempt_id": attempt_id,
                      "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            validate_saved(inst, arm, record)
            write_json_atomic(record_path, record)
            new_arms += 1
            save_progress(last=inst["name"] + "/" + arm)
            print("BİTTİ:", inst["name"], arm, "en iyi", raw["best_cost"], flush=True)
        except Exception as exc:
            write_json_atomic(OUTPUT_ROOT / "errors" / (inst["name"] + "_" + arm + "_" + attempt_id + ".json"),
                              {"instance": inst["name"], "arm": arm, "error": repr(exc),
                               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            save_progress(state="error", last=inst["name"] + "/" + arm)
            raise
    if MAX_NEW_ARMS is not None and new_arms >= MAX_NEW_ARMS:
        break
completed_count = sum(arm_record_path(inst, arm).exists() for inst in INSTANCES for arm in ARMS)
save_progress(state="complete" if completed_count == 48 else "partial")
print(f"Tamamlanan kol: {completed_count}/48. Bu oturumdaki yeni kol: {new_arms}")
"""),
        code("""rows = []
for inst in INSTANCES:
    records = {}
    for arm in ARMS:
        path = arm_record_path(inst, arm)
        if path.exists():
            records[arm] = validate_saved(inst, arm, json.loads(path.read_text()))
    if len(records) == len(ARMS):
        row = {"instance": inst["name"], "group": inst["group"],
               "n": inst["n"], "budget_s": inst["budget_s"],
               "mixed_hardware": len({json.dumps(records[arm]["hardware"], sort_keys=True) for arm in ARMS}) > 1}
        row.update({arm: records[arm]["best_cost"] for arm in ARMS})
        rows.append(row)
table = pd.DataFrame(rows)
summary_dir = OUTPUT_ROOT / "summary"
summary_dir.mkdir(exist_ok=True)
table.to_csv(summary_dir / "complete_instances.csv", index=False)
comparisons = []
if not table.empty and table["mixed_hardware"].any():
    print("UYARI: Bazı örneklerin kolları farklı CPU/kütüphane ortamlarında tamamlandı. Bunları politika seçiminde kullanmayın:",
          table.loc[table["mixed_hardware"], "instance"].tolist())
analysis_table = table.loc[~table["mixed_hardware"]] if not table.empty else table
for treatment in ("beta_promote", "beta_sparse", "frequency_same_gate", "sham"):
    for control in ("native_pop5", "native_pop6", "sham"):
        if treatment == control or analysis_table.empty:
            continue
        difference = 100.0 * (analysis_table[treatment].to_numpy() - analysis_table[control].to_numpy()) / analysis_table[control].to_numpy()
        wins, losses = int((difference < 0).sum()), int((difference > 0).sum())
        comparisons.append({"treatment": treatment, "control": control,
                            "n_instances": len(difference), "wins": wins, "losses": losses,
                            "ties": int((difference == 0).sum()),
                            "mean_paired_pct": float(difference.mean()),
                            "median_paired_pct": float(np.median(difference)),
                            "exploratory_sign_p": float(binomtest(wins, wins + losses).pvalue) if wins + losses else 1.0})
comparison_table = pd.DataFrame(comparisons)
comparison_table.to_csv(summary_dir / "exploratory_comparisons.csv", index=False)
write_json_atomic(summary_dir / "pilot_note.json",
                  {"completed_instances": len(rows), "planned_instances": 8,
                   "same_hardware_instances_for_comparison": len(analysis_table),
                   "interpretation": "Exploratory development pilot on reused v8 cases; no final significance claim.",
                   "next_step": "Freeze one policy before a disjoint new holdout."})
display(table)
display(comparison_table)
"""),
        code("""def checked_zip(name, include_audit):
    local = WORK_ROOT / name
    temp = WORK_ROOT / (name + ".tmp")
    files = []
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for path in sorted(OUTPUT_ROOT.rglob("*")):
            if not path.is_file() or path.suffix == ".zip" or path.name.endswith(".tmp") or path.name.endswith(".tmp.npz"):
                continue
            rel = path.relative_to(OUTPUT_ROOT)
            if not include_audit and ("policy_audit" in rel.parts or path.name in
                                      {"xy.npz", "native.candidates", "live.candidates", "post.candidates", "first.tour", "best.tour", "stdout.txt", "stderr.txt", "policy_arrays.npz"}):
                continue
            archive.write(path, str(rel), compress_type=zipfile.ZIP_STORED if path.suffix == ".npz" else zipfile.ZIP_DEFLATED)
            files.append(str(rel))
    with zipfile.ZipFile(temp) as check:
        assert check.testzip() is None and len(check.namelist()) == len(files)
    os.replace(temp, local)
    target = OUTPUT_ROOT / "archives" / name
    tmp_target = target.with_name(target.name + ".tmp")
    shutil.copy2(local, tmp_target)
    os.replace(tmp_target, target)
    assert sha256_file(local) == sha256_file(target)
    return {"local": str(local), "drive": str(target), "sha256": sha256_file(local),
            "bytes": local.stat().st_size, "files": len(files)}

complete = len(rows) == 8
audit_zip = checked_zip(EXPERIMENT_NAME + "_audit.zip", True) if complete else None
results_zip = checked_zip(EXPERIMENT_NAME + "_results.zip", False)
write_json_atomic(OUTPUT_ROOT / "archives" / "archive_info.json",
                  {"complete": complete, "results": results_zip, "audit": audit_zip})
print("Sonuç ZIP'i:", results_zip)
if complete:
    print("Tam denetim ZIP'i Drive'da:", audit_zip)
    from google.colab import files
    files.download(results_zip["local"])
else:
    print(f"Ara sonuçlar Drive'da; tamamlanan örnek {len(rows)}/8. Yeniden çalıştırınca devam eder.")
"""),
    ]
    for i, cell in enumerate(cells):
        digest = hashlib.sha256("".join(cell["source"]).encode("utf-8")).hexdigest()[:10]
        cell["id"] = f"betatour_{i:02d}_{digest}"
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"colab": {"name": OUTPUT.name, "provenance": []},
                         "kernelspec": {"name": "python3", "display_name": "Python 3"},
                         "language_info": {"name": "python"}, "accelerator": "GPU"},
            "cells": cells}


def main() -> None:
    notebook = make_notebook()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
