# Colab çalışma ortamı: Yüksek RAM ve 12 vCPU (L4 GPU + Yüksek RAM; run v25 ile aynı ortam). GPU kullanılmaz; bu
# ortam türü yalnızca çekirdek sayısı ve bellek için seçilir. Deney bitene kadar aynı ortam türünü kullanın.
EXPERIMENT_NAME = "v26_recomb38"
MAX_NEW_INSTANCES = None      # örn. 16: deneyi kısa oturumlara böler; biten örnekler Drive'dan okunur ve atlanır
SAVE_TO_DRIVE = True
ALLOW_HARDWARE_CHANGE = False  # True: sonraki oturum başka makine türünde sürebilir (rapora not düşülür)
RUN_REFERENCE = True          # sıfırdan GA-EAX referans koşuları (eşit toplam CPU süresi); ana örnekler bitince koşar
REQUIRE_PUSHED_PROTOCOL = True  # kurulum, protokol GitHub'da (main) bu defterin protokolüyle aynı değilse durur

W = 6                         # paralel LKH süreci (tohum) sayısı
SEEDS = list(range(201, 207))        # kayıt eden altı LKH süreci (önceki koşularda 101-106 kullanıldı)
MERGE_SEEDS = list(range(301, 307))  # LKH'nin kendi tur birleştirmesini yapan altı süreç
# (grup, düğüm sayısı, örnek sayısı, aile, üretici tohumu, bütçe s). Tohumların hiçbiri önceki koşularda ya da ön
# denemede kullanılmadı. Bütçe: düğüm başına 12 ms (10.000 düğümde 120 s, önceki koşulardaki gibi).
SYNTHETIC = [("uni10k", 10000, 8, "uniform", 10111, 120.0), ("clu10k", 10000, 8, "clustered", 10212, 120.0),
             ("uni20k", 20000, 3, "uniform", 10313, 240.0), ("clu20k", 20000, 3, "clustered", 10414, 240.0),
             ("uni50k", 50000, 3, "uniform", 10515, 600.0), ("clu50k", 50000, 3, "clustered", 10616, 600.0)]
# Optimumu bilinen TSPLIB örnekleri (EUC_2D); bütçe run v22 ve v25'teki gibi 0,02 x düğüm sayısı saniye.
# tsplib_new: bu fikirle daha önce kullanılmayan beş örnek. tsplib_pilot: pilotun (run v25) beş örneği, yeni tohumlarla.
TSPLIB_SETS = [("tsplib_new", ["pcb3038", "fl3795", "fnl4461", "rl5915", "rl5934"]),
               ("tsplib_pilot", ["rl11849", "usa13509", "brd14051", "d15112", "d18512"])]
K_LIST = 5
SPLIT = 0.5                   # LKH süreçlerinin durdurulduğu an (bütçenin kesri); önceden sabitlenen tek bölme
PER_PROCESS = 17              # "trials" popülasyonunda süreç başına tur: en iyi tur + son 16 atılan deneme turu
EAX_KIDS = 30                 # GA-EAX: ebeveyn çifti başına çocuk sayısı (yazarın önerdiği değer)
EAX_SCRATCH_POP = 300         # sıfırdan GA-EAX referansının popülasyonu (yazarın önerdiği değer)
EAX_NORMAL_MAX_N = 20000      # bu boyuta kadar GA-EAX'in "Normal" sürümü, üstünde "Large" sürümü (bellek)
REFERENCE_GROUPS = ["uni10k", "clu10k", "tsplib_new"]   # sıfırdan GA-EAX referansının koştuğu gruplar
CASCADE_TOURS = 100           # bilinen melez: LKH'nin tek denemelik tur sayısı
CASCADE_GEN_FRACTION = 0.4    # bilinen melezde tur üretimine ayrılan en uzun süre (bütçenin kesri)
CASCADE_SEED0 = 3001
STOP_PAD_S = 2.0              # LKH süreçleri sürelerinden bu kadar sonra dışarıdan durdurulur
REQUIRE_12_VCPUS = True
MIN_RAM_GB = 24.0
REPO_COMMIT = "d63f7e17942a6af5cba14ad010edca1069d477b3"
PROTOCOL_PATH = "experiments/PROTOCOL_v26.md"
PROTOCOL_SHA256 = "__PROTOCOL_SHA256__"    # depodaki protokol dosyasının özeti (defter kurulurken yazılır)
CODE_SHA256_EXPECTED = "__CODE_SHA256__"   # sonraki hücrenin yazdığı byprod_v26.py dosyasının özeti
NOTEBOOK_BUILD = "__NOTEBOOK_BUILD__"      # bu defterin bütün kod hücrelerinin özeti

import os

TEST_MODE = os.environ.get("BETATOUR_V26_TEST") == "1"    # yalnızca yerel öz sınama; Colab'da kapalıdır
if TEST_MODE:
    EXPERIMENT_NAME = "v26_selftest"
    SAVE_TO_DRIVE = False
    REQUIRE_12_VCPUS = False
    REQUIRE_PUSHED_PROTOCOL = False
    MIN_RAM_GB = 0.0
    W = int(os.environ.get("BETATOUR_V26_W", "2"))
    SEEDS, MERGE_SEEDS = SEEDS[:W], MERGE_SEEDS[:W]
    SYNTHETIC = [("uni10k", int(os.environ.get("BETATOUR_V26_N", "1200")), int(os.environ.get("BETATOUR_V26_COUNT", "1")),
                  "uniform", 10111, float(os.environ.get("BETATOUR_V26_BUDGET", "14"))),
                 ("clu50k", 2500, 1, "clustered", 10616, 16.0)]
    # Yerel olarak üretilen TSPLIB biçimli iki dosya; ikincisi (150 düğüm) süreçlerin aynı tura vardığı durumu sınar.
    TSPLIB_SETS = [("tsplib_new", ["test_euc900"]), ("tsplib_pilot", ["test_euc150"])]
    PER_PROCESS = int(os.environ.get("BETATOUR_V26_PER_PROCESS", "5"))
    EAX_SCRATCH_POP = 30
    EAX_NORMAL_MAX_N = 2000                               # 2.500 düğümlü örnek "Large" sürümünü sınar
    REFERENCE_GROUPS = ["uni10k", "tsplib_new"]
    CASCADE_TOURS = int(os.environ.get("BETATOUR_V26_CASCADE_TOURS", "12"))
assert W == len(SEEDS) == len(MERGE_SEEDS) and 0.0 < SPLIT < 1.0 and PER_PROCESS >= 2
