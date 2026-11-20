# Colab çalışma ortamı: GPU (L4 veya A100) + Yüksek RAM (12 vCPU). Deney bitene kadar aynı ortam türünü kullanın.
EXPERIMENT_NAME = "v24_long20k20"
MAX_NEW_INSTANCES = None      # örn. 7: deneyi kısa oturumlara böler; biten örnekler Drive'dan okunur ve atlanır
SAVE_TO_DRIVE = True
ALLOW_HARDWARE_CHANGE = False  # True: sonraki oturum başka GPU türünde sürebilir (rapora not düşülür)

W = 6                         # paralel LKH süreci (tohum) sayısı
SEEDS = list(range(101, 107))
N_NODES = 20000
PER_FAMILY = 10               # 10 uniform + 10 kümeli = 20 yeni örnek
GENERATOR_SEEDS = {"uniform": 8505, "clustered": 8606}   # önceki hiçbir koşuda kullanılmadı
BUDGET_S = 600.0              # run v22'deki 20.000 düğüm bütçesinin (300 s) iki katı
K_GRAPH = 16
K_LIST = 5
K_LEARNED = 3
PAD_SECONDS = 60.0            # dış sınır = bütçe + bütçenin %10'u; yalnızca güvenlik içindir (run v22'de 3 s idi ve
                              # 20.000 düğümde süreçleri, LKH'nin saati bütçeye ulaşmadan durduruyordu)
REQUIRE_12_VCPUS = True
REPO_COMMIT = "d63f7e17942a6af5cba14ad010edca1069d477b3"
CHECKPOINT_SHA256 = "2322f66c752913f6179f8c61f6b50b0f9d21426857d3ec0155e02891fb3ebd70"
ARM_NAMES = ["pop_full", "pop_native", "beta_tree"]
CODE_SHA256_EXPECTED = "__CODE_SHA256__"   # sonraki hücrenin yazdığı alphaenc_v21.py dosyasının özeti (run v21-v23 ile aynı modül)
NOTEBOOK_BUILD = "__NOTEBOOK_BUILD__"      # bu defterin bütün kod hücrelerinin özeti

import os

TEST_MODE = os.environ.get("BETATOUR_V24_TEST") == "1"    # yalnızca yerel öz sınama; Colab'da kapalıdır
if TEST_MODE:
    EXPERIMENT_NAME = "v24_selftest"
    SAVE_TO_DRIVE = False
    REQUIRE_12_VCPUS = False
    W = int(os.environ.get("BETATOUR_V24_W", "2"))
    SEEDS = SEEDS[:W]
    N_NODES = int(os.environ.get("BETATOUR_V24_N", "1000"))
    PER_FAMILY = int(os.environ.get("BETATOUR_V24_PER_FAMILY", "1"))
    BUDGET_S = float(os.environ.get("BETATOUR_V24_BUDGET", "6"))
    PAD_SECONDS = float(os.environ.get("BETATOUR_V24_PAD", "3"))
assert W == len(SEEDS) and K_LEARNED < K_LIST <= K_GRAPH
