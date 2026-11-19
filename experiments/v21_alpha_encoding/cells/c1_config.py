# Colab çalışma ortamı: GPU (L4 veya A100) + Yüksek RAM (12 vCPU). Deney bitene kadar aynı ortam türünü kullanın.
EXPERIMENT_NAME = "v21_alphaenc24"
MAX_NEW_INSTANCES = None      # örn. 8: deneyi kısa oturumlara böler; biten örnekler Drive'dan okunur ve atlanır
SAVE_TO_DRIVE = True
ALLOW_HARDWARE_CHANGE = False  # True: sonraki oturum başka GPU türünde sürebilir (rapora not düşülür)

W = 6                         # paralel LKH süreci (tohum) sayısı
SEEDS = list(range(101, 107))
N_NODES = 10000
PER_FAMILY = 12               # 12 uniform + 12 kümeli = 24 yeni örnek
GENERATOR_SEEDS = {"uniform": 5111, "clustered": 5212}   # önceki hiçbir koşuda kullanılmadı
BUDGET_S = 120.0
K_GRAPH = 16
K_LIST = 5
K_LEARNED = 3
PAD_SECONDS = 3.0
REQUIRE_12_VCPUS = True
REPO_COMMIT = "d63f7e17942a6af5cba14ad010edca1069d477b3"
CHECKPOINT_SHA256 = "2322f66c752913f6179f8c61f6b50b0f9d21426857d3ec0155e02891fb3ebd70"
ARM_NAMES = ["pop_full", "pop_native", "pop_rank", "beta_tree", "beta_rank", "freq_tree"]
CODE_SHA256_EXPECTED = "__CODE_SHA256__"   # sonraki hücrenin yazdığı alphaenc_v21.py dosyasının özeti
NOTEBOOK_BUILD = "__NOTEBOOK_BUILD__"      # bu defterin bütün kod hücrelerinin özeti

import os

TEST_MODE = os.environ.get("BETATOUR_V21_TEST") == "1"    # yalnızca yerel öz sınama; Colab'da kapalıdır
if TEST_MODE:
    EXPERIMENT_NAME = "v21_selftest"
    SAVE_TO_DRIVE = False
    REQUIRE_12_VCPUS = False
    W = int(os.environ.get("BETATOUR_V21_W", "2"))
    SEEDS = SEEDS[:W]
    N_NODES = int(os.environ.get("BETATOUR_V21_N", "1000"))
    PER_FAMILY = int(os.environ.get("BETATOUR_V21_PER_FAMILY", "1"))
    BUDGET_S = float(os.environ.get("BETATOUR_V21_BUDGET", "6"))
assert W == len(SEEDS) and K_LEARNED < K_LIST <= K_GRAPH
