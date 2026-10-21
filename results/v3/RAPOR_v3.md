# TourFold TSP v3 — otomatik rapor

- Tarih: 2026-10-21 05:03
- Ortam: {'nproc': 12, 'ram_gb': 167.1, 'gpu': 'NVIDIA A100-SXM4-80GB, 81920 MiB', 'numpy': '2.1.3'}
- Model: TourFold-v2 (v2 çalıştırmasının kontrol noktası; v3'te yeni eğitim yok). Topluluk 8 tur; 'TourFold-v2(4) 3+POPMUSIC 2' aynı topluluğun yalnız ilk 4 turunu kullanır. Karma listeler: TourFold'un ilk 3 adayı + başka listeden 2 aday; birleşim en çok 8 aday.
- Aşama süreleri (dk, tüm oturumların toplamı): {'1-lkh': 0.1, '1-selftest': 0.0, '3-eval_sets': 2.1, '3-eval_ref': 0.4, '4-train_data': 0.0, '6-train_v2': 0.1, '6-val': 0.0, '7-coverage_v3': 1.0, '8-lkh_eval_v3': 134.0}

## 1. Öz-sınama

Tüm kontroller geçti.

Kullanılan model (v2 eğitiminin son epoch'u; doğrulama, eksik tur kenarı / 1000, ilk 5 aday): M8 = 0.478, M16 = 0.248, α+ascent = 4.560

## 2. Aday listelerinin kalitesi (LKH'siz, referans turlara göre)

TourFold-v2(M) = modelin M turluk toplulukla verdiği sıralama · 3+POPMUSIC 2 = modelin ilk 3 adayı, ardından POPMUSIC listesinden listede olmayan ilk 2 aday · 3+yakın 2 = aynısı en yakın komşularla · ∪ POPMUSIC = modelin ilk 5 adayı + POPMUSIC'in listede olmayan adayları (en çok 8) · sayım+ = topluluk sıklığı · POPMUSIC ve α+ascent = LKH-3'ün aday kümeleri (ilk 5). Referans turlar α-8 grafiğinde bulunduğu için α+ascent satırı biraz iyimserdir.

Eksik tur kenarı / 1000 düğüm (listenin tamamı; küçük = iyi):

| method                      |   val |   uni1k |   clu1k |   uni10k |   clu10k |   tsplib |
|:----------------------------|------:|--------:|--------:|---------:|---------:|---------:|
| TourFold-v2(8) ilk 5        |  0.48 |    0.31 |    0.61 |     0.42 |     0.52 |     2.55 |
| TourFold-v2(4) ilk 5        |  0.87 |    0.58 |    1.27 |     0.74 |     0.94 |     2.99 |
| TourFold-v2(8) 3+POPMUSIC 2 |  0.59 |    1.09 |    1.19 |     1.74 |     2.14 |     3.23 |
| TourFold-v2(4) 3+POPMUSIC 2 |  0.88 |    1.48 |    1.62 |     1.91 |     2.19 |     3.64 |
| TourFold-v2(8) 3+yakın 2    |  1.41 |    1.91 |    2.25 |     2.85 |     3.7  |     4.64 |
| TourFold-v2(8) ∪ POPMUSIC   |  0.2  |    0.16 |    0.31 |     0.29 |     0.29 |     1.95 |
| sayım+ (8)                  |  1.32 |    1.59 |    1.86 |     2.08 |     2.69 |     3.93 |
| POPMUSIC                    |  3.95 |    5.23 |    5.62 |     5.6  |     6.18 |     7.25 |
| α+ascent (LKH-3 varsayılan) |  4.56 |    2.27 |    3.89 |     2.79 |     2.46 |     4.69 |
| en yakın 5 komşu            | 30.52 |   19.53 |   23.41 |    17.51 |    18.23 |    21.68 |

İlk 2 aday içinde kalan tur kenarı oranı (büyük = iyi):

| method                      |    val |   uni1k |   clu1k |   uni10k |   clu10k |   tsplib |
|:----------------------------|-------:|--------:|--------:|---------:|---------:|---------:|
| TourFold-v2(8) ilk 5        | 0.965  |  0.9389 |  0.94   |   0.9275 |   0.9153 |   0.9129 |
| TourFold-v2(4) ilk 5        | 0.9595 |  0.9247 |  0.9252 |   0.9122 |   0.9002 |   0.8992 |
| TourFold-v2(8) 3+POPMUSIC 2 | 0.965  |  0.9389 |  0.94   |   0.9275 |   0.9153 |   0.9129 |
| TourFold-v2(4) 3+POPMUSIC 2 | 0.9595 |  0.9247 |  0.9252 |   0.9122 |   0.9002 |   0.8992 |
| TourFold-v2(8) 3+yakın 2    | 0.965  |  0.9389 |  0.94   |   0.9275 |   0.9153 |   0.9129 |
| TourFold-v2(8) ∪ POPMUSIC   | 0.965  |  0.9389 |  0.94   |   0.9275 |   0.9153 |   0.9129 |
| sayım+ (8)                  | 0.9552 |  0.9426 |  0.9369 |   0.9338 |   0.9232 |   0.9162 |
| POPMUSIC                    | 0.85   |  0.8533 |  0.8451 |   0.8556 |   0.851  |   0.8628 |
| α+ascent (LKH-3 varsayılan) | 0.8866 |  0.8933 |  0.8845 |   0.8964 |   0.8948 |   0.9015 |
| en yakın 5 komşu            | 0.8155 |  0.8256 |  0.8242 |   0.8279 |   0.8293 |   0.8544 |

İlk 3 aday içinde kalan tur kenarı oranı (büyük = iyi):

| method                      |    val |   uni1k |   clu1k |   uni10k |   clu10k |   tsplib |
|:----------------------------|-------:|--------:|--------:|---------:|---------:|---------:|
| TourFold-v2(8) ilk 5        | 0.9966 |  0.9946 |  0.9932 |   0.9918 |   0.9892 |   0.9805 |
| TourFold-v2(4) ilk 5        | 0.9941 |  0.9921 |  0.9903 |   0.9896 |   0.9869 |   0.9772 |
| TourFold-v2(8) 3+POPMUSIC 2 | 0.9966 |  0.9946 |  0.9932 |   0.9918 |   0.9892 |   0.9805 |
| TourFold-v2(4) 3+POPMUSIC 2 | 0.9941 |  0.9921 |  0.9903 |   0.9896 |   0.9869 |   0.9772 |
| TourFold-v2(8) 3+yakın 2    | 0.9966 |  0.9946 |  0.9932 |   0.9918 |   0.9892 |   0.9805 |
| TourFold-v2(8) ∪ POPMUSIC   | 0.9966 |  0.9946 |  0.9932 |   0.9918 |   0.9892 |   0.9805 |
| sayım+ (8)                  | 0.9952 |  0.9937 |  0.9918 |   0.9912 |   0.9889 |   0.9807 |
| POPMUSIC                    | 0.9519 |  0.9487 |  0.9443 |   0.949  |   0.9462 |   0.9542 |
| α+ascent (LKH-3 varsayılan) | 0.9715 |  0.977  |  0.9724 |   0.9768 |   0.977  |   0.9751 |
| en yakın 5 komşu            | 0.9093 |  0.9199 |  0.9167 |   0.9227 |   0.9239 |   0.9369 |

Ortalama liste uzunluğu (aday / düğüm):

| method                      |   val |   uni1k |   clu1k |   uni10k |   clu10k |   tsplib |
|:----------------------------|------:|--------:|--------:|---------:|---------:|---------:|
| TourFold-v2(8) ilk 5        |  5    |    5    |    5    |     5    |     5    |     5    |
| TourFold-v2(4) ilk 5        |  5    |    5    |    5    |     5    |     5    |     5    |
| TourFold-v2(8) 3+POPMUSIC 2 |  5    |    5    |    5    |     5    |     5    |     5    |
| TourFold-v2(4) 3+POPMUSIC 2 |  5    |    5    |    5    |     5    |     5    |     5    |
| TourFold-v2(8) 3+yakın 2    |  5    |    5    |    5    |     5    |     5    |     5    |
| TourFold-v2(8) ∪ POPMUSIC   |  5.58 |    5.69 |    5.73 |     5.74 |     5.76 |     5.86 |
| sayım+ (8)                  |  5    |    5    |    5    |     5    |     5    |     5    |
| POPMUSIC                    |  4.37 |    4.76 |    4.74 |     4.88 |     4.9  |     4.73 |
| α+ascent (LKH-3 varsayılan) |  4.99 |    4.99 |    4.97 |     4.99 |     4.97 |     4.98 |
| en yakın 5 komşu            |  5    |    5    |    5    |     5    |     5    |     5    |

Doğrulama kümesi, dağılıma göre (eksik tur kenarı / 1000):

| method                      |   val-clustered |   val-uniform |
|:----------------------------|----------------:|--------------:|
| TourFold-v2(8) ilk 5        |            0.76 |          0.2  |
| TourFold-v2(4) ilk 5        |            1.38 |          0.36 |
| TourFold-v2(8) 3+POPMUSIC 2 |            0.64 |          0.55 |
| TourFold-v2(4) 3+POPMUSIC 2 |            0.87 |          0.9  |
| TourFold-v2(8) 3+yakın 2    |            1.64 |          1.17 |
| TourFold-v2(8) ∪ POPMUSIC   |            0.29 |          0.12 |
| sayım+ (8)                  |            1.56 |          1.09 |
| POPMUSIC                    |            4.36 |          3.55 |
| α+ascent (LKH-3 varsayılan) |            6.49 |          2.64 |
| en yakın 5 komşu            |           40.3  |         20.81 |

## 3. LKH ile süre–kalite v3 (aynı oturum; toplam süre = topluluk + model + LKH)

Yapılandırmalar:

- **LKH-3 POPMUSIC**: LKH-3'ün kendi POPMUSIC adayları (ascent yok) ve kendi başlangıç turu; taban.
- **Kontrol: POPMUSIC + topluluk(8) başlangıcı**: aynı türden POPMUSIC adayları (topluluk koşusunun ürettiği) + topluluğun en iyi turu başlangıç olarak; model yok. Topluluğun ve başlangıç turunun katkısını ölçer.
- **sayım+ (8)**: topluluk sıklığına göre sıralanmış komşular (genişletilmiş grafik).
- **TourFold-v2(8) ilk 5**: modelin ilk 5 adayı (v2'deki 'TourFold-v2(8)' ile aynı yapılandırma).
- **TourFold-v2(8) 3+POPMUSIC 2**, **TourFold-v2(8) 3+yakın 2**, **TourFold-v2(8) ∪ POPMUSIC**: karma listeler (bölüm 2).
- **TourFold-v2(4) 3+POPMUSIC 2**: aynı karma liste, topluluğun yalnız ilk 4 turuyla (topluluk süresi daha kısa).

Kontrol ve rehberli koşular topluluğun en iyi turundan başlar ('TourFold-v2(4) 3+POPMUSIC 2': ilk 4 turun en iyisi); LKH bu turun kenarlarını aday kümesine kendisi ekler. Topluluk ve modelin süresi bütçeden düşülür. En iyi bilinen: TSPLIB'de optimum; diğerlerinde referans tur ile v2 ve v3 koşularının en iyisi.

Bütçenin %10, %25, %50 ve %100'ünde en iyi bilinene göre fark (%, ortalama; nan = o ana kadar tur yok):

| set    | config                                     |   gap@10%B |   gap@25%B |   gap@50%B |   gap@100%B |   turlu@10%B |   turlu@25%B |   turlu@100%B |   örnek |
|:-------|:-------------------------------------------|-----------:|-----------:|-----------:|------------:|-------------:|-------------:|--------------:|--------:|
| uni1k  | LKH-3 POPMUSIC                             |     0.2241 |     0.0339 |     0.0116 |      0.0098 |            7 |            8 |             8 |       8 |
| uni1k  | Kontrol: POPMUSIC + topluluk(8) başlangıcı |     0.18   |     0.027  |     0.006  |      0.0052 |            3 |            8 |             8 |       8 |
| uni1k  | sayım+ (8)                                 |     0.18   |     0.0415 |     0.0079 |      0.0036 |            3 |            8 |             8 |       8 |
| uni1k  | TourFold-v2(8) ilk 5                       |     0.18   |     0.0401 |     0.0022 |      0.0015 |            3 |            8 |             8 |       8 |
| uni1k  | TourFold-v2(8) 3+POPMUSIC 2                |     0.18   |     0.0362 |     0.0098 |      0.0031 |            3 |            8 |             8 |       8 |
| uni1k  | TourFold-v2(8) 3+yakın 2                   |     0.18   |     0.0416 |     0.0115 |      0.0036 |            3 |            8 |             8 |       8 |
| uni1k  | TourFold-v2(8) ∪ POPMUSIC                  |     0.18   |     0.0514 |     0.0078 |      0.0013 |            3 |            8 |             8 |       8 |
| uni1k  | TourFold-v2(4) 3+POPMUSIC 2                |     0.18   |     0.0281 |     0.009  |      0.0058 |            3 |            8 |             8 |       8 |
| clu1k  | LKH-3 POPMUSIC                             |   nan      |     0.122  |     0.031  |      0.0145 |            0 |            8 |             8 |       8 |
| clu1k  | Kontrol: POPMUSIC + topluluk(8) başlangıcı |   nan      |     0.1758 |     0.0901 |      0.0067 |            0 |            8 |             8 |       8 |
| clu1k  | sayım+ (8)                                 |   nan      |     0.1758 |     0.0886 |      0.036  |            0 |            8 |             8 |       8 |
| clu1k  | TourFold-v2(8) ilk 5                       |   nan      |     0.1758 |     0.0866 |      0.0047 |            0 |            8 |             8 |       8 |
| clu1k  | TourFold-v2(8) 3+POPMUSIC 2                |   nan      |     0.1758 |     0.1031 |      0.0103 |            0 |            8 |             8 |       8 |
| clu1k  | TourFold-v2(8) 3+yakın 2                   |   nan      |     0.1758 |     0.0942 |      0.0468 |            0 |            8 |             8 |       8 |
| clu1k  | TourFold-v2(8) ∪ POPMUSIC                  |   nan      |     0.1758 |     0.0981 |      0.0223 |            0 |            8 |             8 |       8 |
| clu1k  | TourFold-v2(4) 3+POPMUSIC 2                |   nan      |     0.1758 |     0.0398 |      0.0027 |            0 |            8 |             8 |       8 |
| tsplib | LKH-3 POPMUSIC                             |     0.1533 |     0.0787 |     0.0509 |      0.0209 |           11 |           11 |            11 |      11 |
| tsplib | Kontrol: POPMUSIC + topluluk(8) başlangıcı |     0.2987 |     0.0809 |     0.0602 |      0.0508 |           11 |           11 |            11 |      11 |
| tsplib | sayım+ (8)                                 |     0.2878 |     0.0791 |     0.0512 |      0.0268 |           11 |           11 |            11 |      11 |
| tsplib | TourFold-v2(8) ilk 5                       |     0.2747 |     0.0702 |     0.0297 |      0.0234 |           11 |           11 |            11 |      11 |
| tsplib | TourFold-v2(8) 3+POPMUSIC 2                |     0.2965 |     0.0809 |     0.0324 |      0.0247 |           11 |           11 |            11 |      11 |
| tsplib | TourFold-v2(8) 3+yakın 2                   |     0.2924 |     0.0843 |     0.0576 |      0.0475 |           11 |           11 |            11 |      11 |
| tsplib | TourFold-v2(8) ∪ POPMUSIC                  |     0.2969 |     0.0807 |     0.0487 |      0.0222 |           11 |           11 |            11 |      11 |
| tsplib | TourFold-v2(4) 3+POPMUSIC 2                |     0.1754 |     0.062  |     0.048  |      0.0373 |           11 |           11 |            11 |      11 |
| uni10k | LKH-3 POPMUSIC                             |     0.2235 |     0.0638 |     0.0441 |      0.0282 |            8 |            8 |             8 |       8 |
| uni10k | Kontrol: POPMUSIC + topluluk(8) başlangıcı |     0.2984 |     0.1774 |     0.0602 |      0.0393 |            8 |            8 |             8 |       8 |
| uni10k | sayım+ (8)                                 |     0.2984 |     0.125  |     0.0523 |      0.0304 |            8 |            8 |             8 |       8 |
| uni10k | TourFold-v2(8) ilk 5                       |     0.2984 |     0.148  |     0.0751 |      0.0356 |            8 |            8 |             8 |       8 |
| uni10k | TourFold-v2(8) 3+POPMUSIC 2                |     0.2984 |     0.1509 |     0.054  |      0.0311 |            8 |            8 |             8 |       8 |
| uni10k | TourFold-v2(8) 3+yakın 2                   |     0.2984 |     0.1475 |     0.0576 |      0.0325 |            8 |            8 |             8 |       8 |
| uni10k | TourFold-v2(8) ∪ POPMUSIC                  |     0.2984 |     0.155  |     0.0646 |      0.0349 |            8 |            8 |             8 |       8 |
| uni10k | TourFold-v2(4) 3+POPMUSIC 2                |     0.2984 |     0.0956 |     0.0438 |      0.0266 |            8 |            8 |             8 |       8 |
| clu10k | LKH-3 POPMUSIC                             |     0.6136 |     0.0843 |     0.0438 |      0.0249 |            6 |            8 |             8 |       8 |
| clu10k | Kontrol: POPMUSIC + topluluk(8) başlangıcı |     0.4057 |     0.3711 |     0.1084 |      0.0246 |            2 |            8 |             8 |       8 |
| clu10k | sayım+ (8)                                 |     0.4057 |     0.3711 |     0.0582 |      0.0215 |            2 |            8 |             8 |       8 |
| clu10k | TourFold-v2(8) ilk 5                       |     0.4057 |     0.3711 |     0.0873 |      0.0197 |            2 |            8 |             8 |       8 |
| clu10k | TourFold-v2(8) 3+POPMUSIC 2                |     0.4057 |     0.3711 |     0.0847 |      0.0264 |            2 |            8 |             8 |       8 |
| clu10k | TourFold-v2(8) 3+yakın 2                   |     0.4057 |     0.3711 |     0.0768 |      0.0278 |            2 |            8 |             8 |       8 |
| clu10k | TourFold-v2(8) ∪ POPMUSIC                  |     0.4057 |     0.3711 |     0.1064 |      0.0298 |            2 |            8 |             8 |       8 |
| clu10k | TourFold-v2(4) 3+POPMUSIC 2                |     0.4057 |     0.2702 |     0.051  |      0.0238 |            2 |            8 |             8 |       8 |
| uni20k | LKH-3 POPMUSIC                             |     0.1678 |     0.078  |     0.0352 |      0.0145 |            4 |            4 |             4 |       4 |
| uni20k | Kontrol: POPMUSIC + topluluk(8) başlangıcı |     0.2319 |     0.1957 |     0.0941 |      0.055  |            4 |            4 |             4 |       4 |
| uni20k | sayım+ (8)                                 |     0.2319 |     0.1957 |     0.0418 |      0.0149 |            4 |            4 |             4 |       4 |
| uni20k | TourFold-v2(8) ilk 5                       |     0.2319 |     0.1957 |     0.0572 |      0.0288 |            4 |            4 |             4 |       4 |
| uni20k | TourFold-v2(8) 3+POPMUSIC 2                |     0.2319 |     0.1957 |     0.0714 |      0.0422 |            4 |            4 |             4 |       4 |
| uni20k | TourFold-v2(8) 3+yakın 2                   |     0.2319 |     0.1957 |     0.062  |      0.0066 |            4 |            4 |             4 |       4 |
| uni20k | TourFold-v2(8) ∪ POPMUSIC                  |     0.2319 |     0.1957 |     0.0681 |      0.0387 |            4 |            4 |             4 |       4 |
| uni20k | TourFold-v2(4) 3+POPMUSIC 2                |     0.2319 |     0.1079 |     0.0527 |      0.021  |            4 |            4 |             4 |       4 |
| clu20k | LKH-3 POPMUSIC                             |     0.4389 |     0.094  |     0.0364 |      0.0079 |            4 |            4 |             4 |       4 |
| clu20k | Kontrol: POPMUSIC + topluluk(8) başlangıcı |     0.4184 |     0.3669 |     0.0971 |      0.0307 |            3 |            4 |             4 |       4 |
| clu20k | sayım+ (8)                                 |     0.4184 |     0.3669 |     0.0662 |      0.0197 |            3 |            4 |             4 |       4 |
| clu20k | TourFold-v2(8) ilk 5                       |     0.4184 |     0.3669 |     0.0796 |      0.0177 |            3 |            4 |             4 |       4 |
| clu20k | TourFold-v2(8) 3+POPMUSIC 2                |     0.4184 |     0.3669 |     0.0963 |      0.0137 |            3 |            4 |             4 |       4 |
| clu20k | TourFold-v2(8) 3+yakın 2                   |     0.4184 |     0.3669 |     0.0746 |      0.0121 |            3 |            4 |             4 |       4 |
| clu20k | TourFold-v2(8) ∪ POPMUSIC                  |     0.4184 |     0.3669 |     0.1292 |      0.0429 |            3 |            4 |             4 |       4 |
| clu20k | TourFold-v2(4) 3+POPMUSIC 2                |     0.4184 |     0.254  |     0.057  |      0.008  |            3 |            4 |             4 |       4 |

Hedef farka ulaşma süresi (sn, medyan; inf = bütçe içinde ulaşılamadı):

| set    | config                                     |   time_to_1.0% |   time_to_0.5% |   time_to_0.2% |   time_to_0.1% |   time_to_0.05% |
|:-------|:-------------------------------------------|---------------:|---------------:|---------------:|---------------:|----------------:|
| uni1k  | LKH-3 POPMUSIC                             |           0.48 |           0.48 |           0.52 |           0.56 |            0.8  |
| uni1k  | Kontrol: POPMUSIC + topluluk(8) başlangıcı |           0.5  |           0.5  |           0.59 |           0.94 |            1.12 |
| uni1k  | sayım+ (8)                                 |           0.5  |           0.5  |           0.59 |           0.93 |            1.17 |
| uni1k  | TourFold-v2(8) ilk 5                       |           0.5  |           0.5  |           0.59 |           0.97 |            1.17 |
| uni1k  | TourFold-v2(8) 3+POPMUSIC 2                |           0.5  |           0.5  |           0.59 |           0.96 |            1.18 |
| uni1k  | TourFold-v2(8) 3+yakın 2                   |           0.5  |           0.5  |           0.59 |           0.97 |            1.16 |
| uni1k  | TourFold-v2(8) ∪ POPMUSIC                  |           0.5  |           0.5  |           0.59 |           0.97 |            1.25 |
| uni1k  | TourFold-v2(4) 3+POPMUSIC 2                |           0.5  |           0.5  |           0.59 |           0.79 |            0.98 |
| clu1k  | LKH-3 POPMUSIC                             |           0.65 |           0.71 |           0.82 |           1.08 |            1.23 |
| clu1k  | Kontrol: POPMUSIC + topluluk(8) başlangıcı |           0.64 |           0.64 |           1.29 |           1.64 |            2.8  |
| clu1k  | sayım+ (8)                                 |           0.64 |           0.64 |           1.29 |           1.64 |            2.66 |
| clu1k  | TourFold-v2(8) ilk 5                       |           0.64 |           0.64 |           1.29 |           1.64 |            2.89 |
| clu1k  | TourFold-v2(8) 3+POPMUSIC 2                |           0.64 |           0.64 |           1.29 |           1.64 |            2.92 |
| clu1k  | TourFold-v2(8) 3+yakın 2                   |           0.64 |           0.64 |           1.29 |           1.64 |            2.78 |
| clu1k  | TourFold-v2(8) ∪ POPMUSIC                  |           0.64 |           0.64 |           1.29 |           1.64 |            3.01 |
| clu1k  | TourFold-v2(4) 3+POPMUSIC 2                |           0.64 |           0.64 |           1.33 |           1.41 |            1.99 |
| tsplib | LKH-3 POPMUSIC                             |           5.08 |           8.14 |          15.51 |          31.17 |           53.26 |
| tsplib | Kontrol: POPMUSIC + topluluk(8) başlangıcı |           8.26 |           8.26 |          14.96 |          75.01 |          225.84 |
| tsplib | sayım+ (8)                                 |           8.21 |           8.21 |          19.34 |          53.24 |           75.57 |
| tsplib | TourFold-v2(8) ilk 5                       |           8.5  |           8.5  |          14.62 |          30.73 |           56.77 |
| tsplib | TourFold-v2(8) 3+POPMUSIC 2                |           8.56 |           8.56 |          14.53 |          33.79 |           48.42 |
| tsplib | TourFold-v2(8) 3+yakın 2                   |           8.35 |          11.85 |          14.53 |          59.08 |          170.71 |
| tsplib | TourFold-v2(8) ∪ POPMUSIC                  |           9.63 |           9.63 |          16.42 |          55.61 |           74.86 |
| tsplib | TourFold-v2(4) 3+POPMUSIC 2                |           6.36 |           6.41 |          14.53 |          40.45 |           91.9  |
| uni10k | LKH-3 POPMUSIC                             |           9.91 |           9.91 |          12.4  |          20.1  |           39.49 |
| uni10k | Kontrol: POPMUSIC + topluluk(8) başlangıcı |          10.88 |          10.88 |          29.06 |          37.72 |           77.46 |
| uni10k | sayım+ (8)                                 |          10.88 |          10.88 |          27.29 |          36.79 |           66.49 |
| uni10k | TourFold-v2(8) ilk 5                       |          10.88 |          10.88 |          27.16 |          41.84 |           88.4  |
| uni10k | TourFold-v2(8) 3+POPMUSIC 2                |          10.88 |          10.88 |          27.54 |          36    |           69.33 |
| uni10k | TourFold-v2(8) 3+yakın 2                   |          10.88 |          10.88 |          27.59 |          36.59 |           62.89 |
| uni10k | TourFold-v2(8) ∪ POPMUSIC                  |          10.88 |          10.88 |          27.32 |          40.55 |           66.57 |
| uni10k | TourFold-v2(4) 3+POPMUSIC 2                |          10.88 |          10.88 |          19.09 |          28.66 |           51.19 |
| clu10k | LKH-3 POPMUSIC                             |          11.68 |          13.07 |          19.83 |          25.08 |           49    |
| clu10k | Kontrol: POPMUSIC + topluluk(8) başlangıcı |          12.83 |          13.12 |          48.31 |          59.59 |           74.69 |
| clu10k | sayım+ (8)                                 |          12.83 |          13.12 |          46.24 |          52.06 |           62.17 |
| clu10k | TourFold-v2(8) ilk 5                       |          12.83 |          13.12 |          45.55 |          57.24 |           75.81 |
| clu10k | TourFold-v2(8) 3+POPMUSIC 2                |          12.83 |          13.12 |          45.86 |          58.86 |           68.73 |
| clu10k | TourFold-v2(8) 3+yakın 2                   |          12.83 |          13.12 |          46.19 |          55.43 |           68.05 |
| clu10k | TourFold-v2(8) ∪ POPMUSIC                  |          12.83 |          13.12 |          46.66 |          59.21 |           75.53 |
| clu10k | TourFold-v2(4) 3+POPMUSIC 2                |          12.83 |          13.12 |          31.94 |          39.4  |           55.77 |
| uni20k | LKH-3 POPMUSIC                             |          22.66 |          22.66 |          26.12 |          66.09 |          104.01 |
| uni20k | Kontrol: POPMUSIC + topluluk(8) başlangıcı |          26.83 |          26.83 |          71.29 |         125.23 |          inf    |
| uni20k | sayım+ (8)                                 |          26.83 |          26.83 |          71.58 |          96.25 |          143.35 |
| uni20k | TourFold-v2(8) ilk 5                       |          26.83 |          26.83 |          71.66 |          94.67 |          169.2  |
| uni20k | TourFold-v2(8) 3+POPMUSIC 2                |          26.83 |          26.83 |          71.67 |         109.16 |          185.89 |
| uni20k | TourFold-v2(8) 3+yakın 2                   |          26.83 |          26.83 |          71.7  |         115.27 |          178.02 |
| uni20k | TourFold-v2(8) ∪ POPMUSIC                  |          26.83 |          26.83 |          71.82 |         102.8  |          224.93 |
| uni20k | TourFold-v2(4) 3+POPMUSIC 2                |          26.83 |          26.83 |          50.85 |          89.81 |          165.36 |
| clu20k | LKH-3 POPMUSIC                             |          24.5  |          24.5  |          43.73 |          78.64 |          127.59 |
| clu20k | Kontrol: POPMUSIC + topluluk(8) başlangıcı |          28.62 |          33.45 |         113.4  |         139.71 |          215.31 |
| clu20k | sayım+ (8)                                 |          28.62 |          33.45 |         109.94 |         124.8  |          165.39 |
| clu20k | TourFold-v2(8) ilk 5                       |          28.62 |          33.45 |         109.39 |         132.59 |          163.09 |
| clu20k | TourFold-v2(8) 3+POPMUSIC 2                |          28.62 |          33.45 |         111.76 |         125.61 |          170.28 |
| clu20k | TourFold-v2(8) 3+yakın 2                   |          28.62 |          33.45 |         112.55 |         126.83 |          171.59 |
| clu20k | TourFold-v2(8) ∪ POPMUSIC                  |          28.62 |          33.45 |         112.06 |         156.94 |          inf    |
| clu20k | TourFold-v2(4) 3+POPMUSIC 2                |          28.62 |          33.45 |          73.85 |          85.59 |          136.31 |

LKH başlamadan önce harcanan süre (sn, ortalama; topluluk + model + liste; bütçeden düşülür):

| config                                     |   uni1k |   clu1k |   tsplib |   uni10k |   clu10k |   uni20k |   clu20k |
|:-------------------------------------------|--------:|--------:|---------:|---------:|---------:|---------:|---------:|
| LKH-3 POPMUSIC                             |    0    |    0    |     0    |     0    |     0    |     0    |     0    |
| Kontrol: POPMUSIC + topluluk(8) başlangıcı |    1.01 |    2.32 |    27.98 |    26.68 |    43.18 |    80.2  |   104.64 |
| sayım+ (8)                                 |    1.01 |    2.32 |    28.06 |    26.77 |    43.27 |    80.38 |   104.83 |
| TourFold-v2(8) ilk 5                       |    1.07 |    2.38 |    28.2  |    26.91 |    43.41 |    80.65 |   105.11 |
| TourFold-v2(8) 3+POPMUSIC 2                |    1.07 |    2.38 |    28.22 |    26.93 |    43.43 |    80.7  |   105.15 |
| TourFold-v2(8) 3+yakın 2                   |    1.08 |    2.39 |    28.24 |    26.95 |    43.46 |    80.75 |   105.2  |
| TourFold-v2(8) ∪ POPMUSIC                  |    1.07 |    2.38 |    28.2  |    26.92 |    43.42 |    80.67 |   105.13 |
| TourFold-v2(4) 3+POPMUSIC 2                |    0.78 |    1.44 |    17.94 |    17.93 |    26.12 |    50.7  |    63.41 |

Toplu kazanç/beraberlik/kayıp, bütçenin tamamında (K/B/K = yapılandırma rakipten iyi / eşit / kötü; boyut sınıfına göre ve tüm örnekler; p: iki yönlü işaret testi, beraberlikler hariç; p değerleri çoklu karşılaştırma için düzeltilmemiştir):

| yapılandırma                               | rakip                                      | 1k     | 10k    | TSPLIB   | 20k   | tümü     |   p (işaret testi) |
|:-------------------------------------------|:-------------------------------------------|:-------|:-------|:---------|:------|:---------|-------------------:|
| Kontrol: POPMUSIC + topluluk(8) başlangıcı | LKH-3 POPMUSIC                             | 7/6/3  | 6/0/10 | 3/0/8    | 0/0/8 | 16/6/29  |           0.0725   |
| sayım+ (8)                                 | LKH-3 POPMUSIC                             | 7/6/3  | 8/0/8  | 3/0/8    | 3/0/5 | 21/6/24  |           0.766    |
| TourFold-v2(8) ilk 5                       | LKH-3 POPMUSIC                             | 10/4/2 | 7/0/9  | 5/0/6    | 2/0/6 | 24/4/23  |           1        |
| TourFold-v2(8) 3+POPMUSIC 2                | LKH-3 POPMUSIC                             | 9/5/2  | 7/0/9  | 5/1/5    | 2/0/6 | 23/6/22  |           1        |
| TourFold-v2(8) 3+yakın 2                   | LKH-3 POPMUSIC                             | 6/5/5  | 7/0/9  | 3/0/8    | 3/0/5 | 19/5/27  |           0.302    |
| TourFold-v2(8) ∪ POPMUSIC                  | LKH-3 POPMUSIC                             | 9/4/3  | 4/0/12 | 5/1/5    | 1/0/7 | 19/5/27  |           0.302    |
| TourFold-v2(4) 3+POPMUSIC 2                | LKH-3 POPMUSIC                             | 10/4/2 | 8/0/8  | 3/1/7    | 3/0/5 | 24/5/22  |           0.883    |
| sayım+ (8)                                 | Kontrol: POPMUSIC + topluluk(8) başlangıcı | 5/6/5  | 10/0/6 | 8/0/3    | 5/0/3 | 28/6/17  |           0.135    |
| TourFold-v2(8) ilk 5                       | Kontrol: POPMUSIC + topluluk(8) başlangıcı | 8/5/3  | 9/0/7  | 8/1/2    | 6/0/2 | 31/6/14  |           0.0161   |
| TourFold-v2(8) 3+POPMUSIC 2                | Kontrol: POPMUSIC + topluluk(8) başlangıcı | 5/7/4  | 10/0/6 | 8/1/2    | 6/0/2 | 29/8/14  |           0.0315   |
| TourFold-v2(8) 3+yakın 2                   | Kontrol: POPMUSIC + topluluk(8) başlangıcı | 6/4/6  | 8/0/8  | 6/0/5    | 7/0/1 | 27/4/20  |           0.382    |
| TourFold-v2(8) ∪ POPMUSIC                  | Kontrol: POPMUSIC + topluluk(8) başlangıcı | 6/6/4  | 8/0/8  | 9/0/2    | 4/0/4 | 27/6/18  |           0.233    |
| TourFold-v2(4) 3+POPMUSIC 2                | Kontrol: POPMUSIC + topluluk(8) başlangıcı | 7/6/3  | 12/0/4 | 9/0/2    | 8/0/0 | 36/6/9   |           6.57e-05 |
| TourFold-v2(8) 3+POPMUSIC 2                | TourFold-v2(8) ilk 5                       | 2/9/5  | 9/0/7  | 4/2/5    | 5/0/3 | 20/11/20 |           1        |
| TourFold-v2(8) 3+yakın 2                   | TourFold-v2(8) ilk 5                       | 0/10/6 | 7/0/9  | 2/1/8    | 6/0/2 | 15/11/25 |           0.154    |
| TourFold-v2(8) ∪ POPMUSIC                  | TourFold-v2(8) ilk 5                       | 2/11/3 | 7/0/9  | 5/2/4    | 2/0/6 | 16/13/22 |           0.418    |
| TourFold-v2(4) 3+POPMUSIC 2                | TourFold-v2(8) 3+POPMUSIC 2                | 4/9/3  | 11/0/5 | 3/0/8    | 7/0/1 | 25/9/17  |           0.28     |

Tüm örnekler, bütçenin %25, %50 ve %100'ünde (K/B/K ve işaret testi p):

| yapılandırma                               | rakip                                      | K/B/K %25 bütçe   |   p %25 bütçe | K/B/K %50 bütçe   |   p %50 bütçe | K/B/K %100 bütçe   |   p %100 bütçe |
|:-------------------------------------------|:-------------------------------------------|:------------------|--------------:|:------------------|--------------:|:-------------------|---------------:|
| Kontrol: POPMUSIC + topluluk(8) başlangıcı | LKH-3 POPMUSIC                             | 10/1/40           |      2.39e-05 | 13/4/34           |      0.00309  | 16/6/29            |       0.0725   |
| sayım+ (8)                                 | LKH-3 POPMUSIC                             | 10/0/41           |      1.47e-05 | 16/5/30           |      0.0541   | 21/6/24            |       0.766    |
| TourFold-v2(8) ilk 5                       | LKH-3 POPMUSIC                             | 12/0/39           |      0.000198 | 15/4/32           |      0.0186   | 24/4/23            |       1        |
| TourFold-v2(8) 3+POPMUSIC 2                | LKH-3 POPMUSIC                             | 11/1/39           |      9.02e-05 | 15/3/33           |      0.0133   | 23/6/22            |       1        |
| TourFold-v2(8) 3+yakın 2                   | LKH-3 POPMUSIC                             | 10/1/40           |      2.39e-05 | 9/5/37            |      4.06e-05 | 19/5/27            |       0.302    |
| TourFold-v2(8) ∪ POPMUSIC                  | LKH-3 POPMUSIC                             | 9/1/41            |      5.61e-06 | 12/3/36           |      0.000717 | 19/5/27            |       0.302    |
| TourFold-v2(4) 3+POPMUSIC 2                | LKH-3 POPMUSIC                             | 15/2/34           |      0.0094   | 20/3/28           |      0.312    | 24/5/22            |       0.883    |
| sayım+ (8)                                 | Kontrol: POPMUSIC + topluluk(8) başlangıcı | 16/26/9           |      0.23     | 28/10/13          |      0.0275   | 28/6/17            |       0.135    |
| TourFold-v2(8) ilk 5                       | Kontrol: POPMUSIC + topluluk(8) başlangıcı | 16/26/9           |      0.23     | 27/11/13          |      0.0385   | 31/6/14            |       0.0161   |
| TourFold-v2(8) 3+POPMUSIC 2                | Kontrol: POPMUSIC + topluluk(8) başlangıcı | 15/25/11          |      0.557    | 26/10/15          |      0.117    | 29/8/14            |       0.0315   |
| TourFold-v2(8) 3+yakın 2                   | Kontrol: POPMUSIC + topluluk(8) başlangıcı | 13/24/14          |      1        | 26/8/17           |      0.222    | 27/4/20            |       0.382    |
| TourFold-v2(8) ∪ POPMUSIC                  | Kontrol: POPMUSIC + topluluk(8) başlangıcı | 12/26/13          |      1        | 22/8/21           |      1        | 27/6/18            |       0.233    |
| TourFold-v2(4) 3+POPMUSIC 2                | Kontrol: POPMUSIC + topluluk(8) başlangıcı | 35/9/7            |      1.51e-05 | 37/4/10           |      9.85e-05 | 36/6/9             |       6.57e-05 |
| TourFold-v2(8) 3+POPMUSIC 2                | TourFold-v2(8) ilk 5                       | 12/26/13          |      1        | 21/10/20          |      1        | 20/11/20           |       1        |
| TourFold-v2(8) 3+yakın 2                   | TourFold-v2(8) ilk 5                       | 9/27/15           |      0.307    | 20/9/22           |      0.878    | 15/11/25           |       0.154    |
| TourFold-v2(8) ∪ POPMUSIC                  | TourFold-v2(8) ilk 5                       | 7/28/16           |      0.0931   | 17/8/26           |      0.222    | 16/13/22           |       0.418    |
| TourFold-v2(4) 3+POPMUSIC 2                | TourFold-v2(8) 3+POPMUSIC 2                | 34/10/7           |      2.53e-05 | 31/4/16           |      0.04     | 25/9/17            |       0.28     |

Örnek kümesine göre 'TourFold-v2(8) 3+POPMUSIC 2' K/B/K (tümü: v3/results/lkh_eval_v3_wtl.csv):

| set    | at        | Kontrol: POPMUSIC + topluluk(8) başlangıcı   | LKH-3 POPMUSIC   | sayım+ (8)   |
|:-------|:----------|:---------------------------------------------|:-----------------|:-------------|
| clu10k | gap@100%B | 4/0/4                                        | 4/0/4            | 3/0/5        |
| clu10k | gap@25%B  | 0/8/0                                        | 0/0/8            | 0/8/0        |
| clu1k  | gap@100%B | 3/2/3                                        | 5/2/1            | 1/6/1        |
| clu1k  | gap@25%B  | 0/8/0                                        | 2/0/6            | 0/8/0        |
| clu20k | gap@100%B | 3/0/1                                        | 2/0/2            | 4/0/0        |
| clu20k | gap@25%B  | 0/4/0                                        | 0/0/4            | 0/4/0        |
| tsplib | gap@100%B | 8/1/2                                        | 5/1/5            | 6/1/4        |
| tsplib | gap@25%B  | 7/0/4                                        | 6/1/4            | 7/0/4        |
| uni10k | gap@100%B | 6/0/2                                        | 3/0/5            | 4/0/4        |
| uni10k | gap@25%B  | 6/0/2                                        | 0/0/8            | 2/0/6        |
| uni1k  | gap@100%B | 2/5/1                                        | 4/3/1            | 2/5/1        |
| uni1k  | gap@25%B  | 2/1/5                                        | 3/0/5            | 5/1/2        |
| uni20k | gap@100%B | 3/0/1                                        | 0/0/4            | 1/0/3        |
| uni20k | gap@25%B  | 0/4/0                                        | 0/0/4            | 0/4/0        |

v2 ile tekrar karşılaştırması (aynı yapılandırma, aynı örnekler, aynı en iyi bilinen; fark yalnız zamanlama gürültüsünden gelir):

| set    | yapılandırma (v3)    |   v2 gap@25%B |   v3 gap@25%B |   v2 gap@100%B |   v3 gap@100%B |   örnek |
|:-------|:---------------------|--------------:|--------------:|---------------:|---------------:|--------:|
| uni1k  | LKH-3 POPMUSIC       |        0.0339 |        0.0339 |         0.0098 |         0.0098 |       8 |
| clu1k  | LKH-3 POPMUSIC       |        0.1204 |        0.122  |         0.0145 |         0.0145 |       8 |
| tsplib | LKH-3 POPMUSIC       |        0.0776 |        0.0787 |         0.0208 |         0.0209 |      11 |
| uni10k | LKH-3 POPMUSIC       |        0.0611 |        0.0638 |         0.0275 |         0.0282 |       8 |
| clu10k | LKH-3 POPMUSIC       |        0.075  |        0.0843 |         0.0244 |         0.0249 |       8 |
| uni20k | LKH-3 POPMUSIC       |        0.0704 |        0.078  |         0.013  |         0.0145 |       4 |
| clu20k | LKH-3 POPMUSIC       |        0.0858 |        0.094  |         0.0065 |         0.0079 |       4 |
| uni1k  | sayım+ (8)           |        0.0465 |        0.0415 |         0.0036 |         0.0036 |       8 |
| clu1k  | sayım+ (8)           |        0.1758 |        0.1758 |         0.036  |         0.036  |       8 |
| tsplib | sayım+ (8)           |        0.0782 |        0.0791 |         0.0267 |         0.0268 |      11 |
| uni10k | sayım+ (8)           |        0.1078 |        0.125  |         0.0289 |         0.0304 |       8 |
| clu10k | sayım+ (8)           |        0.3711 |        0.3711 |         0.0209 |         0.0215 |       8 |
| uni20k | sayım+ (8)           |        0.1957 |        0.1957 |         0.0142 |         0.0149 |       4 |
| clu20k | sayım+ (8)           |        0.3661 |        0.3669 |         0.0177 |         0.0197 |       4 |
| uni1k  | TourFold-v2(8) ilk 5 |        0.022  |        0.0401 |         0.0006 |         0.0015 |       8 |
| clu1k  | TourFold-v2(8) ilk 5 |        0.1758 |        0.1758 |         0.0064 |         0.0047 |       8 |
| tsplib | TourFold-v2(8) ilk 5 |        0.0647 |        0.0702 |         0.0368 |         0.0234 |      11 |
| uni10k | TourFold-v2(8) ilk 5 |        0.1205 |        0.148  |         0.0235 |         0.0356 |       8 |
| clu10k | TourFold-v2(8) ilk 5 |        0.3711 |        0.3711 |         0.022  |         0.0197 |       8 |
| uni20k | TourFold-v2(8) ilk 5 |        0.1957 |        0.1957 |         0.0333 |         0.0288 |       4 |
| clu20k | TourFold-v2(8) ilk 5 |        0.3661 |        0.3669 |         0.0173 |         0.0177 |       4 |

Topluluk süresi v3/v2 oranı (medyan): 1.038

## Dosyalar

- v3/results: tablolar (csv, json) · v3/figures: grafikler · v3/log_v3.txt: ayrıntılı kayıt
