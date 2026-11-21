# Colab çalışma ortamı: L4 GPU + Yüksek RAM (12 vCPU, 53 GB; run v26 ile aynı ortam). GPU yalnızca BetaTour çıkarımı
# için kullanılır (örnek başına bir kez, 1 s'den az). Deney bitene kadar aynı ortam türünü kullanın.
EXPERIMENT_NAME = "v27_union14"
MAX_NEW_INSTANCES = None      # örn. 7: deneyi kısa oturumlara böler; biten örnekler Drive'dan okunur ve atlanır
SAVE_TO_DRIVE = True
ALLOW_HARDWARE_CHANGE = False  # True: sonraki oturum başka makine türünde sürebilir (rapora not düşülür)
REQUIRE_PUSHED_PROTOCOL = True  # kurulum, protokol GitHub'da (main) bu defterin protokolüyle aynı değilse durur

W = 6                         # paralel LKH süreci (tohum) sayısı
SEEDS = list(range(201, 207))        # kayıt eden altı LKH süreci (run v26 ile aynı)
MERGE_SEEDS = list(range(301, 307))  # LKH'nin kendi tur birleştirmesini yapan altı süreç (run v26 ile aynı)
# (grup, düğüm sayısı, örnek sayısı, aile, üretici tohumu, bütçe s). Tohumların hiçbiri önceki koşularda ya da ön
# sınamada kullanılmadı. Bütçe: düğüm başına 12 ms (run v26 gibi).
SYNTHETIC = [("uni10k", 10000, 6, "uniform", 11111, 120.0), ("clu10k", 10000, 6, "clustered", 11212, 120.0),
             ("uni20k", 20000, 1, "uniform", 11313, 240.0), ("clu20k", 20000, 1, "clustered", 11414, 240.0)]
DECISION_GROUPS = ["uni10k", "clu10k"]   # okumalar (karar kuralı) bu 12 örnek üzerinde; 20.000 düğüm betimseldir
K_LIST = 5
K_GRAPH = 16                  # BetaTour grafiği: 16 en yakın komşu (+ modelin topluluk kenarları), run v21 gibi
SPLIT = 0.5                   # LKH süreçlerinin durdurulduğu an (bütçenin kesri); run v26 gibi
PER_PROCESS = 17              # GA-EAX "trials" popülasyonu: süreç başına en iyi tur + son 16 deneme turu (run v26)
EXTRA_FRACTION = 0.02         # birleşim grafına eklenen kenar sayısı: düğüm sayısının %2'si (model ya da en yakın)
EAX_KIDS = 30                 # GA-EAX: ebeveyn çifti başına çocuk sayısı (yazarın önerdiği değer)
EAX_NORMAL_MAX_N = 20000      # bu boyuta kadar GA-EAX'in "Normal" sürümü, üstünde "Large" sürümü (bellek)
EAX_SCRATCH_POP = 300         # yalnızca öz sınamada (GA-EAX'in duvar saati sınırı denetimi) kullanılır
STOP_PAD_S = 2.0              # LKH süreçleri sürelerinden bu kadar sonra dışarıdan durdurulur
EXACT_PAD_S = 60.0            # tam çözüm süreçleri kendi sınırlarından bu kadar sonra dışarıdan durdurulur (güvenlik)
REQUIRE_12_VCPUS = True
MIN_RAM_GB = 24.0
PYSCIPOPT_VERSION = "6.3.0"   # SCIP 10.0 (pip tekerleğiyle gelir)
REPO_COMMIT = "d63f7e17942a6af5cba14ad010edca1069d477b3"
CHECKPOINT_SHA256 = "2322f66c752913f6179f8c61f6b50b0f9d21426857d3ec0155e02891fb3ebd70"
PROTOCOL_PATH = "experiments/PROTOCOL_v27.md"
PROTOCOL_SHA256 = "__PROTOCOL_SHA256__"    # depodaki protokol dosyasının özeti (defter kurulurken yazılır)
CODE_SHA256_EXPECTED = "__CODE_SHA256__"   # sonraki hücrenin yazdığı byprod_v26.py dosyasının özeti (run v26'nın modülü)
UNION_SHA256_EXPECTED = "__UNION_SHA256__"  # ondan sonraki hücrenin yazdığı unionx_v27.py dosyasının özeti
NOTEBOOK_BUILD = "__NOTEBOOK_BUILD__"      # bu defterin bütün kod hücrelerinin özeti

import os

TEST_MODE = os.environ.get("BETATOUR_V27_TEST") == "1"    # yalnızca yerel öz sınama; Colab'da kapalıdır
if TEST_MODE:
    EXPERIMENT_NAME = "v27_selftest"
    SAVE_TO_DRIVE = False
    REQUIRE_12_VCPUS = False
    REQUIRE_PUSHED_PROTOCOL = False
    MIN_RAM_GB = 0.0
    W = int(os.environ.get("BETATOUR_V27_W", "2"))
    SEEDS, MERGE_SEEDS = SEEDS[:W], MERGE_SEEDS[:W]
    SYNTHETIC = [("uni10k", int(os.environ.get("BETATOUR_V27_N", "1200")), int(os.environ.get("BETATOUR_V27_COUNT", "2")),
                  "uniform", 11111, float(os.environ.get("BETATOUR_V27_BUDGET", "16"))),
                 ("clu20k", 2500, 1, "clustered", 11414, 24.0)]
    DECISION_GROUPS = ["uni10k"]
    PER_PROCESS = int(os.environ.get("BETATOUR_V27_PER_PROCESS", "5"))
    EAX_NORMAL_MAX_N = 2000                               # 2.500 düğümlü örnek "Large" sürümünü sınar
    EAX_SCRATCH_POP = 30
    EXACT_PAD_S = 20.0
assert W == len(SEEDS) == len(MERGE_SEEDS) and 0.0 < SPLIT < 1.0 and PER_PROCESS >= 2 and 0.0 < EXTRA_FRACTION < 1.0
