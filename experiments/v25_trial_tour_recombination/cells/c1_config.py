# Colab çalışma ortamı: Yüksek RAM ve 12 vCPU (L4 ya da A100 GPU + Yüksek RAM). GPU kullanılmaz; bu ortam türü
# yalnızca çekirdek sayısı ve bellek için seçilir. Deney bitene kadar aynı ortam türünü kullanın.
EXPERIMENT_NAME = "v25_trialtours21"
MAX_NEW_INSTANCES = None      # örn. 12: deneyi kısa oturumlara böler; biten örnekler Drive'dan okunur ve atlanır
SAVE_TO_DRIVE = True
ALLOW_HARDWARE_CHANGE = False  # True: sonraki oturum başka makine türünde sürebilir (rapora not düşülür)
RUN_REFERENCE = True          # sıfırdan GA-EAX referans koşuları (eşit toplam CPU süresi); ana örnekler bitince koşar

W = 6                         # paralel LKH süreci (tohum) sayısı
SEEDS = list(range(101, 107))
# (grup, düğüm sayısı, örnek sayısı, aile, üretici tohumu, bütçe s). Tohumların hiçbiri önceki koşularda ya da ön
# denemede kullanılmadı. Bütçe: düğüm başına 12 ms (10.000 düğümde 120 s, önceki koşulardaki gibi).
SYNTHETIC = [("uni10k", 10000, 6, "uniform", 9111, 120.0), ("clu10k", 10000, 6, "clustered", 9212, 120.0),
             ("uni50k", 50000, 2, "uniform", 9515, 600.0), ("clu50k", 50000, 2, "clustered", 9616, 600.0)]
# Optimumu bilinen büyük TSPLIB örnekleri (EUC_2D); bütçe run v22'deki gibi 0,02 x düğüm sayısı saniye.
TSPLIB_NAMES = ["rl11849", "usa13509", "brd14051", "d15112", "d18512"]
K_LIST = 5
SPLITS = [1 / 3, 1 / 2, 2 / 3]  # LKH turlarının alındığı an (bütçenin kesri); kalan süre GA-EAX'e verilir
PRIMARY_SPLIT = 1 / 2         # önceden belirlenen birincil bölme
PER_PROCESS = 17              # süreç başına popülasyona giren tur: en iyi tur + son 16 atılan deneme turu
EAX_KIDS = 30                 # GA-EAX: ebeveyn çifti başına çocuk sayısı (yazarın önerdiği değer)
EAX_SCRATCH_POP = 300         # sıfırdan GA-EAX referansının popülasyonu (yazarın önerdiği değer)
EAX_NORMAL_MAX_N = 20000      # bu boyuta kadar GA-EAX'in "Normal" sürümü, üstünde "Large" sürümü (bellek)
REFERENCE_MAX_N = 20000       # sıfırdan GA-EAX referansı yalnızca bu boyuta kadar koşar
CASCADE_TOURS = 100           # bilinen melez: LKH'nin tek denemelik tur sayısı
CASCADE_GEN_FRACTION = 0.4    # bilinen melezde tur üretimine ayrılan en uzun süre (bütçenin kesri)
CASCADE_SEED0 = 2001
STOP_PAD_S = 2.0              # kayıt eden LKH süreçleri bütçeden bu kadar sonra dışarıdan durdurulur
REQUIRE_12_VCPUS = True
MIN_RAM_GB = 24.0
REPO_COMMIT = "d63f7e17942a6af5cba14ad010edca1069d477b3"
CODE_SHA256_EXPECTED = "__CODE_SHA256__"   # sonraki hücrenin yazdığı byprod_v25.py dosyasının özeti
NOTEBOOK_BUILD = "__NOTEBOOK_BUILD__"      # bu defterin bütün kod hücrelerinin özeti

import os

TEST_MODE = os.environ.get("BETATOUR_V25_TEST") == "1"    # yalnızca yerel öz sınama; Colab'da kapalıdır
if TEST_MODE:
    EXPERIMENT_NAME = "v25_selftest"
    SAVE_TO_DRIVE = False
    REQUIRE_12_VCPUS = False
    MIN_RAM_GB = 0.0
    W = int(os.environ.get("BETATOUR_V25_W", "2"))
    SEEDS = SEEDS[:W]
    SYNTHETIC = [("uni10k", int(os.environ.get("BETATOUR_V25_N", "1200")), 1, "uniform", 9111,
                  float(os.environ.get("BETATOUR_V25_BUDGET", "12"))),
                 ("clu50k", 2500, 1, "clustered", 9616, 15.0)]
    TSPLIB_NAMES = ["test_euc900"]                        # yerel olarak üretilen TSPLIB biçimli dosya
    PER_PROCESS = int(os.environ.get("BETATOUR_V25_PER_PROCESS", "5"))
    EAX_SCRATCH_POP = 30
    EAX_NORMAL_MAX_N = 2000                               # 2.500 düğümlü örnek "Large" sürümünü sınar
    REFERENCE_MAX_N = 2000
    CASCADE_TOURS = int(os.environ.get("BETATOUR_V25_CASCADE_TOURS", "12"))
assert W == len(SEEDS) and PRIMARY_SPLIT in SPLITS and PER_PROCESS >= 2
