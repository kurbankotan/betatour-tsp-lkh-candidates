# Colab çalışma ortamı: GPU (L4 veya A100) + Yüksek RAM (12 vCPU). Deney bitene kadar aynı ortam türünü kullanın.
EXPERIMENT_NAME = "v23_marking12"
MAX_NEW_INSTANCES = None      # örn. 6: deneyi kısa oturumlara böler; biten örnekler Drive'dan okunur ve atlanır
SAVE_TO_DRIVE = True
ALLOW_HARDWARE_CHANGE = False  # True: sonraki oturum başka GPU türünde sürebilir (rapora not düşülür)

W = 6                         # paralel LKH süreci (tohum) sayısı
SEEDS = list(range(101, 107))
N_NODES = 10000
PER_FAMILY = 6                # 6 uniform + 6 kümeli = 12 yeni örnek
GENERATOR_SEEDS = {"uniform": 7111, "clustered": 7212}   # önceki hiçbir koşuda kullanılmadı
MARK_COUNT = 5                # sayım işaretlemesi: altı ilk deneme turunun en az bu kadarında bulunan kenarlar alfa = 0 alır
                              # (model işaretlemesinin eşiği, tohumun ilk turunda aynı sayıda kenarı koruyacak biçimde seçilir)
BUDGET_S = 120.0
K_GRAPH = 16
K_LIST = 5
K_LEARNED = 3
PAD_SECONDS = 3.0
REQUIRE_12_VCPUS = True
REPO_COMMIT = "d63f7e17942a6af5cba14ad010edca1069d477b3"
CHECKPOINT_SHA256 = "2322f66c752913f6179f8c61f6b50b0f9d21426857d3ec0155e02891fb3ebd70"
ARM_NAMES = ["pop_full", "pop_native", "pop_model", "beta_tree", "beta_model", "beta_count"]
CODE_SHA256_EXPECTED = "__CODE_SHA256__"   # alphaenc_v21.py dosyasının özeti (run v21 ve v22 ile aynı modül)
MARKING_SHA256_EXPECTED = "__MARKING_SHA256__"   # marking_v23.py dosyasının özeti (bu koşunun işaretleme modülü)
NOTEBOOK_BUILD = "__NOTEBOOK_BUILD__"      # bu defterin bütün kod hücrelerinin özeti

import os

TEST_MODE = os.environ.get("BETATOUR_V23_TEST") == "1"    # yalnızca yerel öz sınama; Colab'da kapalıdır
if TEST_MODE:
    EXPERIMENT_NAME = "v23_selftest"
    SAVE_TO_DRIVE = False
    REQUIRE_12_VCPUS = False
    W = int(os.environ.get("BETATOUR_V23_W", "2"))
    SEEDS = SEEDS[:W]
    N_NODES = int(os.environ.get("BETATOUR_V23_N", "1000"))
    PER_FAMILY = int(os.environ.get("BETATOUR_V23_PER_FAMILY", "1"))
    BUDGET_S = float(os.environ.get("BETATOUR_V23_BUDGET", "6"))
    MARK_COUNT = min(MARK_COUNT, W)
assert W == len(SEEDS) and K_LEARNED < K_LIST <= K_GRAPH
