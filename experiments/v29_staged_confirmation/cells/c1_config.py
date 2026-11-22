# Colab çalışma ortamı: L4 GPU + Yüksek RAM (12 vCPU, 53 GB; run v26-v28 ile aynı ortam). Bu koşu GPU'yu kullanmaz;
# ortam, önceki koşularla aynı makine türü için seçilir. Deney bitene kadar aynı ortam türünü kullanın.
EXPERIMENT_NAME = "v29_staged23"
MAX_NEW_INSTANCES = None      # örn. 12: deneyi kısa oturumlara böler; biten örnekler Drive'dan okunur ve atlanır
SAVE_TO_DRIVE = True
ALLOW_HARDWARE_CHANGE = False  # True: sonraki oturum başka makine türünde sürebilir (rapora not düşülür)
REQUIRE_PUSHED_PROTOCOL = True  # kurulum, protokol GitHub'da (main) bu defterin protokolüyle aynı değilse durur

W = 6                         # paralel LKH süreci (tohum) sayısı
SEEDS = list(range(201, 207))        # kayıt eden altı LKH süreci (run v26-v28 ile aynı)
MERGE_SEEDS = list(range(301, 307))  # LKH'nin kendi tur birleştirmesini yapan altı süreç (run v26-v28 ile aynı)
# (grup, düğüm sayısı, örnek sayısı, aile, üretici tohumu, bütçe s). Tohumların hiçbiri önceki koşularda ya da ön
# sınamada kullanılmadı. Bütçe: düğüm başına 12 ms (run v26-v28 gibi).
SYNTHETIC = [("uni20k", 20000, 6, "uniform", 13111, 240.0), ("clu20k", 20000, 6, "clustered", 13212, 240.0),
             ("uni50k", 50000, 3, "uniform", 13515, 600.0), ("clu50k", 50000, 3, "clustered", 13616, 600.0)]
# Optimumu bilinen büyük TSPLIB örnekleri (EUC_2D; run v25 ve v26'da da kullanıldı); bütçe run v26'daki gibi
# 0,02 x düğüm sayısı saniye. Betimseldir; hipotez testlerine girmez.
TSPLIB_SETS = [("tsplib", ["rl11849", "usa13509", "brd14051", "d15112", "d18512"])]
DECISION_GROUPS = ["uni20k", "clu20k", "uni50k", "clu50k"]   # hipotez testleri bu 18 sentetik örnek üzerinde
K_LIST = 5
SPLIT = 0.5                   # LKH süreçlerinin durdurulduğu an (bütçenin kesri); run v26-v28 gibi
PER_PROCESS = 17              # süreç başına en iyi tur + son 16 deneme turu: 102 tur (run v26-v28)
GROUPS = 6                    # kademeli kol, aşama 1: aynı anda koşan GA-EAX süreci (grup) sayısı (run v28 gibi)
STAGE1_SHARE = 0.6            # kademeli kol: kalan sürenin aşama 1'e ayrılan kesri (run v28 gibi)
EAX_KIDS = 30                 # GA-EAX: ebeveyn çifti başına çocuk sayısı (yazarın önerdiği değer)
EAX_NORMAL_MAX_N = 20000      # bu boyuta kadar GA-EAX'in "Normal" sürümü, üstünde "Large" sürümü (bellek)
EAX_SCRATCH_POP = 300         # yalnızca öz sınamada (GA-EAX'in duvar saati sınırı denetimi) kullanılır
STOP_PAD_S = 2.0              # LKH süreçleri sürelerinden bu kadar sonra dışarıdan durdurulur
REQUIRE_12_VCPUS = True
MIN_RAM_GB = 24.0
REPO_COMMIT = "d63f7e17942a6af5cba14ad010edca1069d477b3"
PROTOCOL_PATH = "experiments/PROTOCOL_v29.md"
PROTOCOL_SHA256 = "__PROTOCOL_SHA256__"    # depodaki protokol dosyasının özeti (defter kurulurken yazılır)
CODE_SHA256_EXPECTED = "__CODE_SHA256__"   # sonraki hücrenin yazdığı byprod_v26.py dosyasının özeti (run v26'nın modülü)
NOTEBOOK_BUILD = "__NOTEBOOK_BUILD__"      # bu defterin bütün kod hücrelerinin özeti

import os

TEST_MODE = os.environ.get("BETATOUR_V29_TEST") == "1"    # yalnızca yerel öz sınama; Colab'da kapalıdır
if TEST_MODE:
    EXPERIMENT_NAME = "v29_selftest"
    SAVE_TO_DRIVE = False
    REQUIRE_12_VCPUS = False
    REQUIRE_PUSHED_PROTOCOL = False
    MIN_RAM_GB = 0.0
    W = int(os.environ.get("BETATOUR_V29_W", "2"))
    SEEDS, MERGE_SEEDS = SEEDS[:W], MERGE_SEEDS[:W]
    SYNTHETIC = [("uni20k", int(os.environ.get("BETATOUR_V29_N", "1200")), int(os.environ.get("BETATOUR_V29_COUNT", "2")),
                  "uniform", 13111, float(os.environ.get("BETATOUR_V29_BUDGET", "16"))),
                 ("clu50k", 2500, 1, "clustered", 13616, 24.0)]
    TSPLIB_SETS = [("tsplib", ["test_euc900"])]          # yerelde yazılan TSPLIB biçimli dosya; ağ gerekmez
    DECISION_GROUPS = ["uni20k", "clu50k"]
    PER_PROCESS = int(os.environ.get("BETATOUR_V29_PER_PROCESS", "5"))
    GROUPS = int(os.environ.get("BETATOUR_V29_GROUPS", "2"))
    EAX_NORMAL_MAX_N = 2000                               # 2.500 düğümlü örnek "Large" sürümünü sınar
    EAX_SCRATCH_POP = 30
assert W == len(SEEDS) == len(MERGE_SEEDS) and 0.0 < SPLIT < 1.0 and PER_PROCESS >= 2
assert 2 <= GROUPS <= W and W * (PER_PROCESS - 1) >= 2 * GROUPS and 0.0 < STAGE1_SHARE < 1.0
