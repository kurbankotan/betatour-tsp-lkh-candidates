"""Build the self-contained, resumable v11r2 POP6-residual mechanism notebook.

The run-v11 notebook (05_...) is kept unchanged as the record of the 2-vCPU run.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
REPO = EXPERIMENTS.parent
OUTPUT = REPO / "notebooks" / "06_BetaTour_v11r2_POP6_candidate_mechanism.ipynb"
V9_BUILDER = EXPERIMENTS / "v9_live" / "build_colab_notebook.py"
spec = importlib.util.spec_from_file_location("betatour_v9_builder", V9_BUILDER)
if spec is None or spec.loader is None:
    raise RuntimeError("v9 notebook builder missing")
v9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v9)


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"expected exactly one notebook anchor: {old[:70]!r}")
    return source.replace(old, new, 1)


def set_cell(notebook: dict, index: int, source: str) -> None:
    cell = notebook["cells"][index]
    if cell["cell_type"] == "code":
        compile(source, f"<v11 cell {index}>", "exec")
    cell["source"] = source.splitlines(keepends=True)
    cell["id"] = f"betatour_v11_{index:02d}_{hashlib.sha256(source.encode()).hexdigest()[:10]}"


def make_notebook() -> dict:
    notebook = v9.make_notebook()
    old = lambda index: "".join(notebook["cells"][index]["source"])

    set_cell(notebook, 0, """# BetaTour v11r2 — POP6 aday kenar mekanizma pilotu (düzeltilmiş tekrar)

Bu deney **eski v8/v9 geliştirme örneklerinin dört 10k örneğini** kullanır; bağımsız doğrulama değildir. POPMUSIC-6/LKH'nin aynı ilk denemesinden sonra BetaTour, tur sıklığı veya rastgele kontrol, aynı kaynak düğümlere ve aynı sayıda yeni yönsüz aday kenara müdahale eder. Sham aynı model çıkarımını yapar ve aramayı değiştirmez. Native POP6 ve POP7 pratik kıyaslardır. Bütün kol bütçelerine kurulum, model çıkarımı, bekleme ve LKH araması dahildir.

**v11'e göre değişiklikler:** (1) LKH her daha iyi turdan sonra aday satırlarını yeniden düzenlediği için eklenen kenarın 3. sırası artık her deneme sonunda bir *taban* olarak korunur ve kaç kez geri alındığı sayılır. (2) Aynı model girdisini alan kollar ilk saklanan model çıktısıyla karar verir; her kol yine kendi çıkarımını yapar ve süresini öder. (3) Native kolların izlemesi zaman sınırı içinde tur ayrıştırmaz. (4) Kesintide yarım kalmış tur ve olay kayıtları güvenle ele alınır. İlk v11 çalışması T4 ve 2 vCPU üzerinde koştuğu için bu tekrar en az 12 vCPU ister.

**Çalıştırma:** Colab'da **A100 (veya L4) GPU ve Yüksek RAM** seçin; defter 12'den az vCPU görürse durur. Hücreleri sırayla çalıştırın. Tamamlanan kollar `MyDrive/BetaTour/v11r2_pop6_candidate_mechanism` içinde doğrulanarak atlanır; kesilen kol yeniden çalışır. Sonuç ZIP'i sonunda tarayıcıya otomatik iner, tam denetim ZIP'i Drive'da kalır.

**Tahmini süre:** kurulum ve LKH derlemesi ≈ 3–5 dk; ısınma ve doğrulama ≈ 1 dk; dört örnek × altı kol × 120 sn = **48 dk** çözücü; kayıt ve arşiv ≈ 2–4 dk. Toplam ≈ 55–60 dk.
""")

    config = old(1)
    config = replace_once(config, 'EXPERIMENT_NAME = "v9_live_pilot"',
                          'EXPERIMENT_NAME = "v11r2_pop6_candidate_mechanism"')
    config = replace_once(config,
        'ARMS = ["native_pop5", "native_pop6", "sham", "beta_promote", "beta_sparse", "frequency_same_gate"]',
        'ARMS = ["native_pop6", "native_pop7", "sham", "random_add", "frequency_add", "beta_add"]')
    config = replace_once(config,
        '    ["uni20k", 20000, 2, "uniform", 4309, 300.0],\n    ["clu20k", 20000, 2, "clustered", 4410, 300.0],\n', '')
    config = replace_once(config, 'SPARSE_FRACTION = 0.25',
                          'SPARSE_FRACTION = 0.25\nGRAPH_K = 16')
    config = replace_once(config, 'assert len(SEEDS) == 6 and len(EXPECTED_XY_SHA256) == 8',
        'EXPECTED_XY_SHA256 = {name: digest for name, digest in EXPECTED_XY_SHA256.items()\n'
        '                      if "10k" in name}\n'
        '# Time-to-target costs: run-v10 native POPMUSIC-6 best of six (A100, 12 vCPU).\n'
        'TARGET_COSTS = {"augment_clu10k_000": 33760199,\n'
        '                "augment_clu10k_001": 16201186,\n'
        '                "augment_uni10k_000": 72087781,\n'
        '                "augment_uni10k_001": 71648570}\n'
        'assert len(SEEDS) == 6 and len(EXPECTED_XY_SHA256) == 4 and len(TARGET_COSTS) == 4')
    set_cell(notebook, 1, config)

    setup = old(2)
    setup = replace_once(setup, 'betatour_v9_live_work', 'betatour_v11r2_candidate_work')
    setup = replace_once(setup,
        '"python": platform.python_version(), "torch": torch.__version__}',
        '"python": platform.python_version(), "torch": torch.__version__,\n'
        '            "numpy": np.__version__, "scipy": scipy.__version__,\n'
        '            "pandas": pd.__version__, "cuda": torch.version.cuda}')
    set_cell(notebook, 2, setup)

    files = {
        "livehook/coordinator.py": EXPERIMENTS / "livehook" / "coordinator.py",
        "v11_candidate/BetaTourSearchHook.h": HERE / "BetaTourSearchHook.h",
        "v11_candidate/BetaTourVisit.h": HERE / "BetaTourVisit.h",
        "v11_candidate/build_search_hook.py": HERE / "build_search_hook.py",
        "v11_candidate/policy.py": HERE / "policy.py",
        "v11_candidate/pilot.py": HERE / "pilot.py",
        "v11_candidate/run_cell.py": HERE / "run_cell.py",
        "v11_candidate/analysis_cell.py": HERE / "analysis_cell.py",
        "v11_candidate/build_colab_notebook.py": HERE / "build_colab_notebook.py",
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
sys.path[:0] = [str(CODE_ROOT / "livehook"), str(CODE_ROOT / "v11_candidate")]
import build_search_hook
from coordinator import read_tour
from pilot import run_guided_arm, run_native_arm, sha256_file
from policy import POLICY_VERSION
print("Dondurulmuş deney kodu:", CODE_SHA256)
""")

    build = old(4)
    build = replace_once(build, 'PATCHED_DIR = LKH_ROOT / "LKH-3.0.14-BetaTour"',
                         'PATCHED_DIR = LKH_ROOT / "LKH-3.0.14-BetaTour-v11r2"')
    build = replace_once(build, 'build_livehook.build(SOURCE_DIR, PATCHED_DIR, jobs=min(6, CPU_COUNT))',
                         'build_search_hook.build(SOURCE_DIR, PATCHED_DIR, jobs=min(6, CPU_COUNT))')
    build = replace_once(build,
        'if not SOURCE_INFO_PATH.exists():',
        'for filename, expected in build_search_hook.UPSTREAM_SHA256.items():\n'
        '    assert hashlib.sha256((SOURCE_DIR / "SRC" / filename).read_bytes()).hexdigest() == expected, filename\n'
        'if not SOURCE_INFO_PATH.exists():')
    build = replace_once(build,
        'BUILD_INFO = {"native_sha256": sha256_file(NATIVE_LKH),',
        'patch_manifest = json.loads((PATCHED_DIR / "betatour_build.json").read_text())\n'
        'assert patch_manifest["upstream_sha256"] == build_search_hook.UPSTREAM_SHA256\n'
        'assert patch_manifest["search_hook_sha256"] == CODE_SHA256["v11_candidate/BetaTourSearchHook.h"]\n'
        'assert patch_manifest["visit_header_sha256"] == CODE_SHA256["v11_candidate/BetaTourVisit.h"]\n'
        'assert patch_manifest["binary_sha256"] == sha256_file(PATCHED_LKH)\n'
        'BUILD_INFO = {"native_sha256": sha256_file(NATIVE_LKH),')
    build = replace_once(build,
        '"upstream_findtour_sha256": LKH_FINDTOUR_SHA256}',
        '"upstream_findtour_sha256": LKH_FINDTOUR_SHA256,\n'
        '              "action_slot_rule": patch_manifest["action_slot_rule"],\n'
        '              "candidate_reordering": patch_manifest["upstream_candidate_reordering"]}')
    build = replace_once(build, 'print("LKH 3.0.14 derlendi:", BUILD_INFO)',
        'print("LKH 3.0.14 derlendi:", {k: v for k, v in BUILD_INFO.items() if k != "candidate_reordering"})\n'
        'print("LKH aday sırası yeniden düzenleme kuralı (kaynak denetimi):",\n'
        '      json.dumps(BUILD_INFO["candidate_reordering"], indent=1))')
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
            "time_to_target_costs_v10_native_pop6_best_of_six": TARGET_COSTS,
            "policy_version": POLICY_VERSION,
            "sparse_fraction": SPARSE_FRACTION, "graph_k": GRAPH_K,
            "repo_commit": REPO_COMMIT, "checkpoint_sha256": CHECKPOINT_SHA256,
            "hardware": HARDWARE, "code_sha256": CODE_SHA256, "build": BUILD_INFO,
            "analysis_plan": "Mechanism pilot on reused cases; no paper performance claim before frozen independent holdout."}
manifest_path = OUTPUT_ROOT / "manifest.json"
if manifest_path.exists():
    saved = json.loads(manifest_path.read_text())
    for key in MANIFEST:
        if key != "hardware":
            assert saved[key] == MANIFEST[key], f"Drive'daki pilotun {key} ayarı değişti; yeni deney adı kullanın."
    for key in ("gpu", "vcpus"):
        assert saved["hardware"][key] == HARDWARE[key], f"Donanım {key} değişti; eşli pilotu aynı türde çalıştırın."
else:
    write_json_atomic(manifest_path, MANIFEST)

# GPU warmup and exact first-trial/native equivalence check.
warm_xy = T.to_int_coords(T.gen_instance(100, "uniform", np.random.default_rng(9090)))
warm_succ = np.stack([np.roll(np.arange(100), -1)] * 6, axis=1).astype(np.int32)
M.predict_tourfold(MODEL, warm_xy, GRAPH_K, DEVICE,
                   ens_succ=warm_succ, ens_cost=np.ones(6))
warm_problem = WORK_ROOT / "warmup.tsp"
T.write_tsp(warm_problem, warm_xy)
warm_dir = WORK_ROOT / "warmup" / uuid.uuid4().hex
warm_dir.parent.mkdir(exist_ok=True)
warm_cache = WORK_ROOT / "warmup" / ("cache_" + uuid.uuid4().hex)
warm = run_guided_arm("sham", PATCHED_LKH, warm_problem, warm_xy, warm_dir,
                      SEEDS, 15.0, MODEL, DEVICE, T, M,
                      fraction=SPARSE_FRACTION, graph_k=GRAPH_K,
                      decision_cache_dir=warm_cache, model_identity=CHECKPOINT_SHA256)
warm_guided_dir = WORK_ROOT / "warmup" / uuid.uuid4().hex
warm_guided = run_guided_arm("beta_add", PATCHED_LKH, warm_problem, warm_xy, warm_guided_dir,
                             SEEDS, 10.0, MODEL, DEVICE, T, M,
                             fraction=SPARSE_FRACTION, graph_k=GRAPH_K,
                             decision_cache_dir=warm_cache, model_identity=CHECKPOINT_SHA256)
assert warm["policy_audit"]["decision_basis"] == "fresh_inference"
assert warm_guided["policy_audit"]["decision_basis"] == "cached_inference", "Ayrı kolların model girdisi eşleşmedi"
assert warm_guided["first_tour_hashes"] == warm["first_tour_hashes"], "Ayrı kolların ilk turu eşleşmedi"
assert sum(warm_guided["policy_audit"]["selected_actions_per_seed"]) > 0, "Isınma kolunda eylem seçilmedi"
for seed in SEEDS:
    hook_keys = (warm_guided_dir / f"seed_{seed}" / "telemetry.txt").read_text()
    assert "slot_floor_enforced 1" in hook_keys and "slot_restorations" in hook_keys
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
print("GPU, POP6 yaması ve ilk tur eşitliği doğrulandı.")
""")

    instances = old(6)
    instances = replace_once(instances, 'assert len(INSTANCES) == 8',
                             'assert len(INSTANCES) == 4')
    set_cell(notebook, 6, instances)
    set_cell(notebook, 7, (HERE / "run_cell.py").read_text())
    set_cell(notebook, 8, (HERE / "analysis_cell.py").read_text())

    archive = old(9)
    archive = replace_once(archive,
        '{"xy.npz", "native.candidates", "live.candidates", "post.candidates", "first.tour", "best.tour", "stdout.txt", "stderr.txt", "policy_arrays.npz"}',
        '{"xy.npz", "native.candidates", "live.candidates", "post.candidates", "first.tour", "best.tour", "action.txt", "stdout.txt", "stderr.txt", "policy_arrays.npz"}')
    archive = replace_once(archive, '"policy_audit" in rel.parts',
                           '"policy_audit" in rel.parts or "decision_cache" in rel.parts')
    archive = replace_once(archive, 'complete = len(rows) == 8',
                           'complete = len(rows) == len(INSTANCES)')
    archive = replace_once(archive,
        'tamamlanan örnek {len(rows)}/8.',
        'tamamlanan örnek {len(rows)}/{len(INSTANCES)}.')
    set_cell(notebook, 9, archive)

    notebook["metadata"]["colab"]["name"] = OUTPUT.name
    return notebook


def main() -> None:
    notebook = make_notebook()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
