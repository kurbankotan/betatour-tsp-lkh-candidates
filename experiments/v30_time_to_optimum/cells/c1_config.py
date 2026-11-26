from google.colab import drive; drive.mount("/content/drive")  # noqa: E702
# Run v30: kanıtlı optimuma ulaşma süresi (protokol: experiments/PROTOCOL_v30.md).
# Ortam: L4 GPU + Yüksek RAM (12 vCPU) önerilir; T4 + 8 vCPU da kabul edilir (süre uzar). GPU yalnızca BetaTour
# çıkarımı içindir. Bütün oturumlar aynı makine türünde olmalıdır (süre LKH-3'ün işlemci saatiyle ölçülür).
# Tahmini süre: kurulum 3-5 dk, öz sınama 1-2 dk, ilk denemeler ve listeler 4-10 dk, ana adım 12 vCPU'da 3-5 saat
# (8 vCPU'da 4-7 saat), analiz ve arşiv 1-2 dk. Oturum düşerse defteri baştan çalıştırın; biten koşular atlanır.
import os

EXPERIMENT_NAME = "v30_tto37"
SAVE_TO_DRIVE = True
REQUIRE_PUSHED_PROTOCOL = True     # kurulum, protokol ve defter GitHub'da (main) bu kurulumla aynı değilse durur
ALLOW_HARDWARE_CHANGE = False      # True: sonraki oturum başka makine türünde sürebilir (rapora not düşülür)

SEEDS = list(range(201, 207))      # altı tohum (çalışmadaki gibi); ilk denemeler ve bütün kollar
K_LIST = 5
K_UNION = 10
K_LEARNED = 3                      # mix: BetaTour'un ilk üçü (run v21-v23'teki listeler)
K_GRAPH = 16                       # BetaTour grafı: 16 en yakın komşu + topluluk kenarları
CAP_PER_NODE = 0.05                # üst sınır: düğüm başına LKH-3 işlemci saniyesi (TOTAL_TIME_LIMIT)
PAR_FACTOR = 10.0                  # PAR10: sınırda optimuma ulaşmayan koşu 10 x sınır sayılır
TIE_FRACTION = 0.01                # iki kolun PAR10 farkı sınırın %1'ini aşmıyorsa eşit sayılır (protokol)
JOB_ORDER_SEED = 3030              # bütün koşuların tek rastgele sırası (yük kollara eşit dağılır)
BOOTSTRAP_SEED = 30
ARMS = ["pop", "alpha", "pop_r", "beta", "count", "mix", "union"]

# Pilotun 18 sentetik örneği (certified_pilot): (grup, düğüm, üretilen örnek sayısı, aile, üretici tohumu, kullanılan
# sıra numaraları). Optimumlar pilotta Concorde ile kanıtlandı; dosyaların SHA-256 değerleri aynı kodla hesaplandı.
SYNTHETIC = [("uni1k", 1000, 5, "uniform", 15101, [0, 1, 2, 3, 4]),
             ("clu1k", 1000, 5, "clustered", 15202, [0, 1, 2, 3, 4]),
             ("uni2k", 2000, 5, "uniform", 15303, [1, 3, 4]),
             ("clu2k", 2000, 5, "clustered", 15404, [0, 1, 2, 3, 4])]
OPTIMA = {"cp1_uni1k_00": 22893008, "cp1_uni1k_01": 23299083, "cp1_uni1k_02": 23243281, "cp1_uni1k_03": 22488252,
          "cp1_uni1k_04": 23364800, "cp1_clu1k_00": 10139325, "cp1_clu1k_01": 9698285, "cp1_clu1k_02": 12844846,
          "cp1_clu1k_03": 15289751, "cp1_clu1k_04": 16172229, "cp1_uni2k_01": 32773730, "cp1_uni2k_03": 32335931,
          "cp1_uni2k_04": 32618292, "cp1_clu2k_00": 14235210, "cp1_clu2k_01": 16900417, "cp1_clu2k_02": 12243261,
          "cp1_clu2k_03": 20185266, "cp1_clu2k_04": 11765714}
PROBLEM_SHA256 = {
    "cp1_uni1k_00": "87568f46d4a0965410c71c64ccc25f3fabd1524a677e8a1c86707d102e6da676",
    "cp1_uni1k_01": "36cc1821f033b5fe1030eeb2a2639cb8b5b242d9218ac0902c1a96fc866aa507",
    "cp1_uni1k_02": "811fcba04fb23c7fd9d31f00ef88670008701543c028b55e6336008d7031c155",
    "cp1_uni1k_03": "2448de1fdaa8a6243969d9ae67c3da09959b1c9ea275f94da8de1184918b3844",
    "cp1_uni1k_04": "cca760a0929c4c1ab2783a86976b5ff6ec4d690a623bfb0b8d90e396136e366a",
    "cp1_clu1k_00": "916c16a3b8e2377cd422c117dcccdb551276e243a143fb425d831a0887c299e7",
    "cp1_clu1k_01": "dfc4393271f2b226bd22426bda15d2b31a58895cad5c7936aadd4c52f325535d",
    "cp1_clu1k_02": "b8270b97b147ebc58b9e781a8a7fc9598bf7737d26ce4ca4cd26dd8d3d50b264",
    "cp1_clu1k_03": "b40f02b2bdf15ed6381fb51cac2d7ac9812ea057b8d8aecef9d611104c93b02e",
    "cp1_clu1k_04": "0db90d52e2339909d5752027860adda7250e4e42c4d6cf67b520477d256b34b4",
    "cp1_uni2k_01": "883d2cb398a5dd03ced216cb0db1d652c552948ec4fc20d88fb24a92d138736c",
    "cp1_uni2k_03": "4c186483f84d7a2e382fad65fa51ddd0d1a24a1998ca21d0cb1027faaf7d5201",
    "cp1_uni2k_04": "53a3c28a260cc9f644e14d2ab7798d4734500f45102868e3bfda9ea6bf5cc034",
    "cp1_clu2k_00": "136d3c107f07c81c3ffb95793f448f3076ab5e08c93ad846eaa00e484c3e0a9a",
    "cp1_clu2k_01": "66bb41052e43b2de6077ba790860403c3c00047258ba075aea8dc864bc44d1da",
    "cp1_clu2k_02": "76bd375638cf2e25c368f01e9afd2b2df95448c10caa374b3833c8870fb91dec",
    "cp1_clu2k_03": "dba165ed5003f8db5478b3e2789605e3e13c6fa8ce6b3167cbe89234efc773d2",
    "cp1_clu2k_04": "a55e5d2875a1ae459c5d3dd7e2e5779a69d8f7f6e91ffe8167e8f8f842804a90"}
# Yayımlanmış optimumu olan TSPLIB örnekleri (EUC_2D, 1.002-2.392 düğüm); optimumlar tsplab.TSPLIB_OPT'tan.
TSPLIB_NAMES = ["pr1002", "u1060", "vm1084", "pcb1173", "d1291", "rl1304", "rl1323", "nrw1379", "fl1400", "u1432",
                "fl1577", "d1655", "vm1748", "u1817", "rl1889", "d2103", "u2152", "u2319", "pr2392"]
PILOT_DIR = "/content/drive/MyDrive/BetaTour/certpilot1"   # isteğe bağlı: pilotun optimal turları (kapsama analizi)
TSPLIB_TEST_OPTIMA = None          # yalnızca test kipinde: yerelde yazılan TSPLIB biçimli dosyaların optimumları
SELF_N = 200

REPO_COMMIT = "d63f7e17942a6af5cba14ad010edca1069d477b3"
CHECKPOINT_SHA256 = "2322f66c752913f6179f8c61f6b50b0f9d21426857d3ec0155e02891fb3ebd70"
PROTOCOL_PATH = "experiments/PROTOCOL_v30.md"
NOTEBOOK_FILE = "notebooks/33_BetaTour_v30_time_to_optimum.ipynb"
PROTOCOL_SHA256 = "__PROTOCOL_SHA256__"        # protokol dosyasının özeti (defter kurulurken yazılır)
ALPHAENC_SHA256 = "__ALPHAENC_SHA256__"        # alphaenc_v21.py (run v21'in modülü, değiştirilmeden)
TTO_SHA256 = "__TTO_SHA256__"                  # tto_v30.py (bu koşunun modülü)
NOTEBOOK_BUILD = "__NOTEBOOK_BUILD__"          # bütün kod hücrelerinin, modüllerin ve protokolün özeti

TEST_MODE = os.environ.get("BETATOUR_V30_TEST") == "1"     # yalnızca yerel öz sınama; Colab'da kapalıdır
if TEST_MODE:
    EXPERIMENT_NAME = "v30_selftest"
    SAVE_TO_DRIVE = False
    REQUIRE_PUSHED_PROTOCOL = False
    import json as _json
    _test = _json.loads(open(os.environ["BETATOUR_V30_TEST_SPEC"]).read())
    SYNTHETIC = [tuple(x) for x in _test["synthetic"]]
    OPTIMA, PROBLEM_SHA256 = _test["optima"], _test["sha256"]
    TSPLIB_NAMES = _test["tsplib_names"]
    TSPLIB_TEST_OPTIMA = _test["tsplib_optima"]
    CAP_PER_NODE = float(_test.get("cap_per_node", CAP_PER_NODE))
    PILOT_DIR = _test.get("pilot_dir", "")
    SELF_N = 60
assert len(SEEDS) == 6 and K_LEARNED < K_LIST < K_UNION <= K_GRAPH
