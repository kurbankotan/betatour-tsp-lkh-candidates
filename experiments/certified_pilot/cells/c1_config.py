from google.colab import drive; drive.mount("/content/drive")  # noqa: E702
# Kanıtlı aday grafı: karar pilotu. BetaTour çalışmasının bir koşusu değildir; yeni projenin ilk denemesidir.
# Ortam: L4 GPU + Yüksek RAM (12 vCPU, 53 GB) önerilir. GPU yalnızca BetaTour çıkarımı içindir (CPU'da da çalışır);
# Concorde ve LKH süreçleri vCPU sayısı kadar paralel koşar. 2 vCPU'lu standart ortamda süre birkaç kat uzar.
# Tahmini süre (12 vCPU): kurulum 5-8 dk, öz sınama 1-2 dk, LKH 2-4 dk, BetaTour 1 dk, Concorde 30-60 dk, analiz 2-4 dk.
import os

EXPERIMENT_NAME = "certpilot1"
SAVE_TO_DRIVE = True            # sonuçlar Drive'da: MyDrive/BetaTour/certpilot1 (oturum düşerse biten işler atlanır)

# (grup, düğüm sayısı, örnek sayısı, aile, üretici tohumu); tohumlar önceki koşularda kullanılmadı.
SYNTHETIC = [("uni1k", 1000, 5, "uniform", 15101), ("clu1k", 1000, 5, "clustered", 15202),
             ("uni2k", 2000, 5, "uniform", 15303), ("clu2k", 2000, 5, "clustered", 15404)]
TSPLIB_NAMES = ["pr1002", "pcb1173", "rl1304", "u1817"]   # optimumu bilinen; Concorde'un bulduğu optimum buna eşit olmalı
FIRST_TRIAL_SEEDS = list(range(201, 207))   # BetaTour'un okuduğu altı ilk deneme turu (POPMUSIC-5; çalışmadaki gibi)
K_LIST = 5                      # aday listesi uzunluğu
K_GRAPH = 16                    # BetaTour grafiği: 16 en yakın komşu + topluluk kenarları (çalışmadaki gibi)
RANK_DEPTH = 50                 # en yakın komşu ve alfa sıralamalarının uzunluğu
UB_TIME_PER_NODE = 0.01         # üst sınır (ÜS) için LKH koşusu: düğüm başına saniye
K_RC = 200                      # concorde -z: deneyde düğüm başına yazılan en küçük indirgenmiş maliyet sayısı
K_RC_SELF = 60                  # öz sınamada kısmi dökümün sınandığı değer (tam dökümle karşılaştırılır)
RC_TOL = 1e-3                   # indirgenmiş maliyet karşılaştırmalarında pay (maliyet birimi; yazım hassasiyeti eklenir)
CONCORDE_SEED = 99
ROOT_TIMEOUT_S = 1800           # kök LP koşusu için üst süre
FULL_TIMEOUT_S = 2700           # tam çözüm (kanıtlı optimum) için üst süre; aşılırsa örnek optimumsuz raporlanır
SELF_N = 200                    # öz sınama örneğinin düğüm sayısı
REPO_COMMIT = "d63f7e17942a6af5cba14ad010edca1069d477b3"     # run v27-v29 ile aynı: tsplab, tsp_model, betatour_v2.pt
CHECKPOINT_SHA256 = "2322f66c752913f6179f8c61f6b50b0f9d21426857d3ec0155e02891fb3ebd70"
CODE_SHA256_EXPECTED = "__CODE_SHA256__"    # sonraki hücrenin yazdığı certpilot.py dosyasının özeti
NOTEBOOK_BUILD = "__NOTEBOOK_BUILD__"       # bu defterin bütün kod hücrelerinin özeti

# Karar kuralı (pilottan önce yazıldı; README'deki ile aynı):
# G1  C2 sertifikası, kanıtlı optimumu olan her örnekte denetimleri geçer.
# G2  C2 ve ÜS = LKH turu ile kalan grafın ortalama derecesi her sentetik grupta en çok G2_MAX_MEAN_DEGREE
#     (kısmi dökümde karar verilemeyen düğümlerin payı en çok %1).
# G3  5'li listesi kanıtla tam olan düğümlerin payı (C2, ÜS = LKH turu): BetaTour, hem LKH'nin alfa-5 listesini hem
#     sayma listesini her iki boyutta (1.000 ve 2.000 düğüm; tekdüze ve kümeli birlikte) en az G3_MARGIN_PP puan aşar.
G2_MAX_MEAN_DEGREE = 8.0
G3_MARGIN_PP = 5.0

TEST_MODE = os.environ.get("CERTPILOT_TEST") == "1"     # yalnızca yerel öz sınama; Colab'da kapalıdır
if TEST_MODE:
    EXPERIMENT_NAME = "certpilot_selftest"
    SAVE_TO_DRIVE = False
    SYNTHETIC = [("uni1k", 60, 2, "uniform", 15101), ("clu1k", 70, 1, "clustered", 15202),
                 ("uni2k", 80, 1, "uniform", 15303), ("clu2k", 90, 1, "clustered", 15404)]
    TSPLIB_NAMES = ["test_euc60"]
    SELF_N = 40
    K_RC = 30
    K_RC_SELF = 12
    RANK_DEPTH = 30
    ROOT_TIMEOUT_S = FULL_TIMEOUT_S = 600
