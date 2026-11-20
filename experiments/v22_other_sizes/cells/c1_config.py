# Colab çalışma ortamı: GPU (L4 veya A100) + Yüksek RAM (12 vCPU). Deney bitene kadar aynı ortam türünü kullanın.
EXPERIMENT_NAME = "v22_sizes43"
MAX_NEW_INSTANCES = None      # örn. 15: deneyi kısa oturumlara böler; biten örnekler Drive'dan okunur ve atlanır
SAVE_TO_DRIVE = True
ALLOW_HARDWARE_CHANGE = False  # True: sonraki oturum başka GPU türünde sürebilir (rapora not düşülür)

W = 6                         # paralel LKH süreci (tohum) sayısı
SEEDS = list(range(101, 107))
# (grup, düğüm sayısı, örnek sayısı, aile, üretici tohumu, bütçe s). Tohumların hiçbiri önceki koşularda kullanılmadı.
SYNTHETIC = [("uni1k", 1000, 8, "uniform", 6101, 5.0), ("clu1k", 1000, 8, "clustered", 6202, 5.0),
             ("uni20k", 20000, 8, "uniform", 6505, 300.0), ("clu20k", 20000, 8, "clustered", 6606, 300.0)]
# Önceki koşuların 11 TSPLIB örneği; bütçe max(10, 0,02 x düğüm sayısı) saniye.
TSPLIB_NAMES = ["pcb3038", "fl3795", "fnl4461", "rl5915", "rl5934", "pla7397",
                "rl11849", "usa13509", "brd14051", "d15112", "d18512"]
K_GRAPH = 16
K_LIST = 5
K_LEARNED = 3
PAD_SECONDS = 3.0
REQUIRE_12_VCPUS = True
REPO_COMMIT = "d63f7e17942a6af5cba14ad010edca1069d477b3"
CHECKPOINT_SHA256 = "2322f66c752913f6179f8c61f6b50b0f9d21426857d3ec0155e02891fb3ebd70"
ARM_NAMES = ["pop_full", "pop_native", "beta_tree", "freq_tree"]
CODE_SHA256_EXPECTED = "__CODE_SHA256__"   # sonraki hücrenin yazdığı alphaenc_v21.py dosyasının özeti (run v21 ile aynı modül)
NOTEBOOK_BUILD = "__NOTEBOOK_BUILD__"      # bu defterin bütün kod hücrelerinin özeti

import os

TEST_MODE = os.environ.get("BETATOUR_V22_TEST") == "1"    # yalnızca yerel öz sınama; Colab'da kapalıdır
if TEST_MODE:
    EXPERIMENT_NAME = "v22_selftest"
    SAVE_TO_DRIVE = False
    REQUIRE_12_VCPUS = False
    W = int(os.environ.get("BETATOUR_V22_W", "2"))
    SEEDS = SEEDS[:W]
    SYNTHETIC = [("uni1k", 600, 1, "uniform", 6101, 5.0), ("clu20k", 1500, 1, "clustered", 6606, 8.0)]
    TSPLIB_NAMES = ["test_euc800", "test_ceil700"]       # yerel olarak üretilen iki TSPLIB biçimli dosya
assert W == len(SEEDS) and K_LEARNED < K_LIST <= K_GRAPH
