# TourFold TSP v2 — otomatik rapor

- Tarih: 2026-10-20 23:34
- Ortam: {'nproc': 12, 'ram_gb': 53.0, 'gpu': 'NVIDIA L4, 23034 MiB', 'numpy': '2.1.3'}
- Veri: 30000 eğitim örneği (n ∈ [100, 500]), topluluk 16 tur; v2 ayarları: grafiğe en çok 4 topluluk kenarı/düğüm, eğitimde topluluk boyu [4, 8, 12, 16], değerlendirmede M = 8 ve 16
- Aşama süreleri (dk, tüm oturumların toplamı): {'1-lkh': 0.1, '1-selftest': 0.0, '2-profile': 0.0, '3-eval_sets': 2.1, '3-eval_ref': 0.5, '3-classic_coverage': 0.0, '4-train_data': 0.0, '5-v1_tourfold': 1.2, '5-v1_single': 0.2, '6-train_v2': 104.6, '7-decision_v2': 2.4, '8-lkh_eval_v2': 128.7}

## 1. Öz-sınama

Tüm kontroller geçti.

## 2. TourFold-v2 eğitimi (doğrulama; eksik tur kenarı / 1000, ilk 5 aday)

|   epoch |   son_kayip |    M8 |   M16 |   alpha_ascent |
|--------:|------------:|------:|------:|---------------:|
|      12 |       1.592 | 0.478 | 0.248 |           4.56 |

## 3. Karar deneyi v2

Yöntemler: TourFold-v2 (M) = yeni grafik ve M turluk toplulukla · TourFold-v1 = ilk çalıştırmanın modeli · tek-örnek = topluluksuz model · sayım (kNN) = en yakın 16 komşu üzerinde topluluk sıklığı · sayım+ (M) = aynı sıklık, topluluk kenarlarıyla genişletilmiş grafikte · alpha+ascent = LKH-3 varsayılan adayları · POPMUSIC = LKH'nin ascent'siz adayları. Referans turlar α-8 grafiğinde bulunduğu için alpha+ascent sütunu biraz iyimserdir.

Eksik tur kenarı / 1000 düğüm (ilk 5 aday; küçük = iyi):

| method             |   clu10k |   clu1k |   tsplib |   uni10k |   uni1k |   val |
|:-------------------|---------:|--------:|---------:|---------:|--------:|------:|
| POPMUSIC           |     6.18 |    5.62 |     7.25 |     5.6  |    5.23 |  3.95 |
| TourFold-v1 (M=16) |     0.82 |    4.56 |     4.1  |     0.34 |    0.25 |  7.23 |
| TourFold-v2 (M=16) |     0.24 |    0.38 |     2.22 |     0.22 |    0.16 |  0.25 |
| TourFold-v2 (M=8)  |     0.51 |    0.59 |     2.53 |     0.44 |    0.34 |  0.48 |
| alpha+ascent       |     2.46 |    3.89 |     4.69 |     2.79 |    2.27 |  4.56 |
| sayım (kNN)        |     1.84 |    5    |     4.39 |     1.12 |    0.72 |  7.6  |
| sayım+ (M=16)      |     1.4  |    0.78 |     2.59 |     1.14 |    0.66 |  0.66 |
| sayım+ (M=8)       |     2.69 |    1.86 |     3.93 |     2.08 |    1.59 |  1.32 |
| tek-örnek          |     0.89 |    5.09 |     4.12 |     0.44 |    0.44 |  7.81 |

Doğrulama kümesi, dağılıma göre:

| method             |   val-clustered |   val-uniform |
|:-------------------|----------------:|--------------:|
| POPMUSIC           |            4.36 |          3.55 |
| TourFold-v1 (M=16) |           14.41 |          0.12 |
| TourFold-v2 (M=16) |            0.41 |          0.09 |
| TourFold-v2 (M=8)  |            0.75 |          0.2  |
| alpha+ascent       |            6.49 |          2.64 |
| sayım (kNN)        |           14.7  |          0.55 |
| sayım+ (M=16)      |            0.78 |          0.54 |
| sayım+ (M=8)       |            1.56 |          1.09 |
| tek-örnek          |           15.35 |          0.32 |

İlk 2 aday içinde kalan tur kenarı oranı (büyük = iyi):

| method             |   clu10k |   clu1k |   tsplib |   uni10k |   uni1k |    val |
|:-------------------|---------:|--------:|---------:|---------:|--------:|-------:|
| POPMUSIC           |   0.851  |  0.8451 |   0.8628 |   0.8556 |  0.8533 | 0.85   |
| TourFold-v1 (M=16) |   0.9216 |  0.9388 |   0.9173 |   0.9339 |  0.9467 | 0.9624 |
| TourFold-v2 (M=16) |   0.9252 |  0.9425 |   0.9195 |   0.9362 |  0.9477 | 0.9682 |
| TourFold-v2 (M=8)  |   0.9152 |  0.94   |   0.9131 |   0.9275 |  0.939  | 0.965  |
| alpha+ascent       |   0.8948 |  0.8845 |   0.9015 |   0.8964 |  0.8933 | 0.8866 |
| sayım (kNN)        |   0.9281 |  0.9379 |   0.9179 |   0.9376 |  0.9451 | 0.9518 |
| sayım+ (M=16)      |   0.9281 |  0.9393 |   0.9178 |   0.9375 |  0.945  | 0.9564 |
| sayım+ (M=8)       |   0.9232 |  0.9369 |   0.9162 |   0.9338 |  0.9426 | 0.9552 |
| tek-örnek          |   0.9012 |  0.8877 |   0.8919 |   0.9076 |  0.9035 | 0.8823 |

Kenar sabitleme (kesinlik/duyarlılık kenar düzeyinde toplanarak). Süzülmüş omurga eşiği τ = 0.98, yalnız olasılık eşiği τ = 0.98 (doğrulamada kesinlik ≥ 0.999 hedefi):

| group   | method                      |   precision |   recall |   fixed_frac |
|:--------|:----------------------------|------------:|---------:|-------------:|
| clu10k  | TourFold-v2 olasılığı (M=8) |      1      |   0.0003 |       0.0003 |
| clu10k  | omurga (M=16)               |      0.9967 |   0.6241 |       0.6262 |
| clu10k  | omurga (M=8)                |      0.9914 |   0.6835 |       0.6894 |
| clu10k  | omurga ∩ TourFold-v2 (M=8)  |      1      |   0.0003 |       0.0003 |
| clu1k   | TourFold-v2 olasılığı (M=8) |      0.9974 |   0.0538 |       0.0539 |
| clu1k   | omurga (M=16)               |      0.9972 |   0.684  |       0.6859 |
| clu1k   | omurga (M=8)                |      0.993  |   0.7339 |       0.739  |
| clu1k   | omurga ∩ TourFold-v2 (M=8)  |      0.9974 |   0.0536 |       0.0537 |
| tsplib  | TourFold-v2 olasılığı (M=8) |      0.9969 |   0.1419 |       0.1423 |
| tsplib  | omurga (M=16)               |      0.9971 |   0.6399 |       0.6417 |
| tsplib  | omurga (M=8)                |      0.9904 |   0.6922 |       0.6988 |
| tsplib  | omurga ∩ TourFold-v2 (M=8)  |      0.9983 |   0.1406 |       0.1408 |
| uni10k  | TourFold-v2 olasılığı (M=8) |      0.9985 |   0.0164 |       0.0164 |
| uni10k  | omurga (M=16)               |      0.9977 |   0.6433 |       0.6448 |
| uni10k  | omurga (M=8)                |      0.9928 |   0.7051 |       0.7102 |
| uni10k  | omurga ∩ TourFold-v2 (M=8)  |      0.9985 |   0.0164 |       0.0164 |
| uni1k   | TourFold-v2 olasılığı (M=8) |      0.9978 |   0.0929 |       0.0931 |
| uni1k   | omurga (M=16)               |      0.9982 |   0.6832 |       0.6844 |
| uni1k   | omurga (M=8)                |      0.9942 |   0.7359 |       0.7403 |
| uni1k   | omurga ∩ TourFold-v2 (M=8)  |      0.998  |   0.0926 |       0.0928 |
| val     | TourFold-v2 olasılığı (M=8) |      0.9964 |   0.3232 |       0.3244 |
| val     | omurga (M=16)               |      0.9981 |   0.7711 |       0.7725 |
| val     | omurga (M=8)                |      0.9949 |   0.8063 |       0.8104 |
| val     | omurga ∩ TourFold-v2 (M=8)  |      0.997  |   0.3203 |       0.3212 |

### Hüküm (TourFold-v2, M = 8)

- **clu10k**: TourFold-v2 0.51 · TourFold-v2 (M=16): 0.24 (K/B/K 0/0/8) · sayım+ (M=8): 2.69 (K/B/K 8/0/0) · sayım (kNN): 1.84 (K/B/K 8/0/0) · POPMUSIC: 6.17 (K/B/K 8/0/0) · alpha+ascent: 2.46 (K/B/K 8/0/0) · TourFold-v1 (M=16): 0.83 (K/B/K 6/1/1) · tek-örnek: 0.89 (K/B/K 7/0/1)
- **clu1k**: TourFold-v2 0.59 · TourFold-v2 (M=16): 0.38 (K/B/K 1/52/11) · sayım+ (M=8): 1.86 (K/B/K 35/27/2) · sayım (kNN): 5.00 (K/B/K 58/5/1) · POPMUSIC: 5.62 (K/B/K 63/1/0) · alpha+ascent: 3.89 (K/B/K 59/4/1) · TourFold-v1 (M=16): 4.56 (K/B/K 57/6/1) · tek-örnek: 5.09 (K/B/K 58/5/1)
- **tsplib**: TourFold-v2 2.53 · TourFold-v2 (M=16): 2.22 (K/B/K 2/12/17) · sayım+ (M=8): 3.93 (K/B/K 26/3/2) · sayım (kNN): 4.39 (K/B/K 24/5/2) · POPMUSIC: 7.25 (K/B/K 30/1/0) · alpha+ascent: 4.69 (K/B/K 25/1/5) · TourFold-v1 (M=16): 4.10 (K/B/K 20/6/5) · tek-örnek: 4.12 (K/B/K 21/5/5)
- **uni10k**: TourFold-v2 0.44 · TourFold-v2 (M=16): 0.23 (K/B/K 0/0/8) · sayım+ (M=8): 2.08 (K/B/K 8/0/0) · sayım (kNN): 1.12 (K/B/K 7/0/1) · POPMUSIC: 5.60 (K/B/K 8/0/0) · alpha+ascent: 2.79 (K/B/K 8/0/0) · TourFold-v1 (M=16): 0.34 (K/B/K 2/1/5) · tek-örnek: 0.44 (K/B/K 5/2/1)
- **uni1k**: TourFold-v2 0.34 · TourFold-v2 (M=16): 0.16 (K/B/K 0/53/11) · sayım+ (M=8): 1.59 (K/B/K 39/24/1) · sayım (kNN): 0.72 (K/B/K 23/35/6) · POPMUSIC: 5.23 (K/B/K 63/1/0) · alpha+ascent: 2.27 (K/B/K 54/8/2) · TourFold-v1 (M=16): 0.25 (K/B/K 6/47/11) · tek-örnek: 0.44 (K/B/K 17/35/12)
- **val**: TourFold-v2 0.48 · TourFold-v2 (M=16): 0.25 (K/B/K 1/944/55) · sayım+ (M=8): 1.32 (K/B/K 164/831/5) · sayım (kNN): 7.60 (K/B/K 490/492/18) · POPMUSIC: 3.95 (K/B/K 642/340/18) · alpha+ascent: 4.56 (K/B/K 636/338/26) · TourFold-v1 (M=16): 7.23 (K/B/K 447/527/26) · tek-örnek: 7.81 (K/B/K 499/475/26)

## 4. LKH ile süre–kalite v2 (aynı oturum; toplam süre = topluluk + model + LKH)

Rehberli koşularda başlangıç turu topluluğun en iyi turudur; LKH bu turun kenarlarını aday kümesine kendisi ekler. Sabit kenarlı koşularda LKH'nin bildirdiği maliyetler gerçek tur uzunluğuna çevrildi. LKH-3 varsayılan, bütçe dolunca durdurulur (ascent bitmemişse turu yoktur).

Bütçenin %10, %25, %50 ve %100'ünde en iyi bilinene göre fark (%, ortalama; nan = o ana kadar tur yok):

| set    | config                           |   gap@10%B |   gap@25%B |   gap@50%B |   gap@100%B |   turlu@10%B |   turlu@25%B |   turlu@100%B |   örnek |
|:-------|:---------------------------------|-----------:|-----------:|-----------:|------------:|-------------:|-------------:|--------------:|--------:|
| uni1k  | LKH-3 POPMUSIC (ascent yok)      |     0.2341 |     0.0339 |     0.0116 |      0.0098 |            8 |            8 |             8 |       8 |
| uni1k  | LKH-3 varsayılan (α+ascent)      |   nan      |     0.1163 |     0.0146 |      0.0076 |            0 |            5 |             8 |       8 |
| uni1k  | Topluluk(8) + sayım+             |     0.2232 |     0.0465 |     0.0079 |      0.0036 |            4 |            8 |             8 |       8 |
| uni1k  | Topluluk(8) omurgası sabit       |     0.2232 |     0.0258 |     0.0255 |      0.0254 |            4 |            8 |             8 |       8 |
| uni1k  | TourFold-v2(8)                   |     0.2232 |     0.022  |     0.0058 |      0.0006 |            4 |            8 |             8 |       8 |
| uni1k  | TourFold-v2(16)                  |     0.2232 |     0.0708 |     0.0046 |      0.0028 |            4 |            8 |             8 |       8 |
| uni1k  | TourFold-v2(8) + süzülmüş omurga |     0.2232 |     0.0393 |     0.0052 |      0.0044 |            4 |            8 |             8 |       8 |
| clu1k  | LKH-3 POPMUSIC (ascent yok)      |   nan      |     0.1204 |     0.031  |      0.0145 |            0 |            8 |             8 |       8 |
| clu1k  | LKH-3 varsayılan (α+ascent)      |   nan      |   nan      |     0.5591 |      0.2771 |            0 |            0 |             8 |       8 |
| clu1k  | Topluluk(8) + sayım+             |   nan      |     0.1758 |     0.0885 |      0.036  |            0 |            8 |             8 |       8 |
| clu1k  | Topluluk(8) omurgası sabit       |   nan      |     0.1758 |     0.0751 |      0.0371 |            0 |            8 |             8 |       8 |
| clu1k  | TourFold-v2(8)                   |   nan      |     0.1758 |     0.0914 |      0.0063 |            0 |            8 |             8 |       8 |
| clu1k  | TourFold-v2(16)                  |   nan      |     0.1758 |     0.1194 |      0.0374 |            0 |            8 |             8 |       8 |
| clu1k  | TourFold-v2(8) + süzülmüş omurga |   nan      |     0.1758 |     0.0933 |      0.008  |            0 |            8 |             8 |       8 |
| tsplib | LKH-3 POPMUSIC (ascent yok)      |     0.1509 |     0.0776 |     0.0508 |      0.0208 |           11 |           11 |            11 |      11 |
| tsplib | LKH-3 varsayılan (α+ascent)      |   nan      |     1.4234 |     0.4278 |      0.2165 |            0 |            2 |             7 |      11 |
| tsplib | Topluluk(8) + sayım+             |     0.2788 |     0.0782 |     0.051  |      0.0267 |           11 |           11 |            11 |      11 |
| tsplib | Topluluk(8) omurgası sabit       |     0.2765 |     0.0985 |     0.087  |      0.0829 |           11 |           11 |            11 |      11 |
| tsplib | TourFold-v2(8)                   |     0.2737 |     0.0647 |     0.0471 |      0.0368 |           11 |           11 |            11 |      11 |
| tsplib | TourFold-v2(16)                  |     0.324  |     0.1136 |     0.0306 |      0.023  |           11 |           11 |            11 |      11 |
| tsplib | TourFold-v2(8) + süzülmüş omurga |     0.2714 |     0.0708 |     0.0541 |      0.0485 |           11 |           11 |            11 |      11 |
| uni10k | LKH-3 POPMUSIC (ascent yok)      |     0.1782 |     0.0611 |     0.0441 |      0.0275 |            8 |            8 |             8 |       8 |
| uni10k | LKH-3 varsayılan (α+ascent)      |   nan      |   nan      |   nan      |    nan      |            0 |            0 |             0 |       8 |
| uni10k | Topluluk(8) + sayım+             |     0.2758 |     0.1078 |     0.0505 |      0.0289 |            8 |            8 |             8 |       8 |
| uni10k | Topluluk(8) omurgası sabit       |     0.2758 |     0.0925 |     0.0493 |      0.0424 |            8 |            8 |             8 |       8 |
| uni10k | TourFold-v2(8)                   |     0.2758 |     0.1205 |     0.0486 |      0.0235 |            8 |            8 |             8 |       8 |
| uni10k | TourFold-v2(16)                  |     0.2758 |     0.2075 |     0.0814 |      0.0306 |            8 |            8 |             8 |       8 |
| uni10k | TourFold-v2(8) + süzülmüş omurga |     0.2758 |     0.113  |     0.0512 |      0.0269 |            8 |            8 |             8 |       8 |
| clu10k | LKH-3 POPMUSIC (ascent yok)      |     0.5149 |     0.075  |     0.0431 |      0.0244 |            6 |            8 |             8 |       8 |
| clu10k | LKH-3 varsayılan (α+ascent)      |   nan      |   nan      |   nan      |    nan      |            0 |            0 |             0 |       8 |
| clu10k | Topluluk(8) + sayım+             |     0.4339 |     0.3711 |     0.0556 |      0.0209 |            6 |            8 |             8 |       8 |
| clu10k | Topluluk(8) omurgası sabit       |     0.4339 |     0.3711 |     0.0665 |      0.0499 |            6 |            8 |             8 |       8 |
| clu10k | TourFold-v2(8)                   |     0.4339 |     0.3711 |     0.0557 |      0.022  |            6 |            8 |             8 |       8 |
| clu10k | TourFold-v2(16)                  |     0.4339 |     0.3711 |     0.3133 |      0.0281 |            6 |            8 |             8 |       8 |
| clu10k | TourFold-v2(8) + süzülmüş omurga |     0.4339 |     0.3711 |     0.0598 |      0.018  |            6 |            8 |             8 |       8 |
| uni20k | LKH-3 POPMUSIC (ascent yok)      |     0.1464 |     0.0629 |     0.0248 |      0.0055 |            4 |            4 |             4 |       4 |
| uni20k | LKH-3 varsayılan (α+ascent)      |   nan      |   nan      |   nan      |    nan      |            0 |            0 |             0 |       4 |
| uni20k | Topluluk(8) + sayım+             |     0.2244 |     0.1882 |     0.0314 |      0.0067 |            4 |            4 |             4 |       4 |
| uni20k | Topluluk(8) omurgası sabit       |     0.2244 |     0.1882 |     0.0455 |      0.032  |            4 |            4 |             4 |       4 |
| uni20k | TourFold-v2(8)                   |     0.2244 |     0.1882 |     0.0537 |      0.0259 |            4 |            4 |             4 |       4 |
| uni20k | TourFold-v2(16)                  |     0.2244 |     0.1882 |     0.1053 |      0.0338 |            4 |            4 |             4 |       4 |
| uni20k | TourFold-v2(8) + süzülmüş omurga |     0.2244 |     0.1882 |     0.051  |      0.0178 |            4 |            4 |             4 |       4 |
| clu20k | LKH-3 POPMUSIC (ascent yok)      |     0.3306 |     0.0858 |     0.0349 |      0.0065 |            4 |            4 |             4 |       4 |
| clu20k | LKH-3 varsayılan (α+ascent)      |   nan      |   nan      |   nan      |    nan      |            0 |            0 |             0 |       4 |
| clu20k | Topluluk(8) + sayım+             |     0.4743 |     0.3661 |     0.062  |      0.0177 |            4 |            4 |             4 |       4 |
| clu20k | Topluluk(8) omurgası sabit       |     0.4743 |     0.3661 |     0.064  |      0.0454 |            4 |            4 |             4 |       4 |
| clu20k | TourFold-v2(8)                   |     0.4743 |     0.3661 |     0.0706 |      0.0173 |            4 |            4 |             4 |       4 |
| clu20k | TourFold-v2(16)                  |     0.4743 |     0.3661 |     0.2822 |      0.0351 |            4 |            4 |             4 |       4 |
| clu20k | TourFold-v2(8) + süzülmüş omurga |     0.4743 |     0.3661 |     0.0623 |      0.0087 |            4 |            4 |             4 |       4 |

Hedef farka ulaşma süresi (sn, medyan; inf = bütçe içinde ulaşılamadı):

| set    | config                           |   time_to_1.0% |   time_to_0.5% |   time_to_0.2% |   time_to_0.1% |   time_to_0.05% |
|:-------|:---------------------------------|---------------:|---------------:|---------------:|---------------:|----------------:|
| uni1k  | LKH-3 POPMUSIC (ascent yok)      |           0.48 |           0.48 |           0.51 |           0.56 |            0.8  |
| uni1k  | LKH-3 varsayılan (α+ascent)      |           1.23 |           1.23 |           1.25 |           1.53 |            1.58 |
| uni1k  | Topluluk(8) + sayım+             |           0.5  |           0.5  |           0.6  |           0.98 |            1.23 |
| uni1k  | Topluluk(8) omurgası sabit       |           0.5  |           0.5  |           0.6  |           0.97 |            1.1  |
| uni1k  | TourFold-v2(8)                   |           0.5  |           0.5  |           0.6  |           0.99 |            1.15 |
| uni1k  | TourFold-v2(16)                  |           0.5  |           0.5  |           0.6  |           0.99 |            1.68 |
| uni1k  | TourFold-v2(8) + süzülmüş omurga |           0.5  |           0.5  |           0.6  |           0.99 |            1.17 |
| clu1k  | LKH-3 POPMUSIC (ascent yok)      |           0.64 |           0.69 |           0.8  |           1.06 |            1.21 |
| clu1k  | LKH-3 varsayılan (α+ascent)      |           1.73 |           1.81 |           2.44 |           2.44 |            2.46 |
| clu1k  | Topluluk(8) + sayım+             |           0.61 |           0.61 |           1.26 |           1.61 |            2.64 |
| clu1k  | Topluluk(8) omurgası sabit       |           0.61 |           0.61 |           1.26 |           1.57 |            2.53 |
| clu1k  | TourFold-v2(8)                   |           0.61 |           0.61 |           1.26 |           1.61 |            2.92 |
| clu1k  | TourFold-v2(16)                  |           0.61 |           0.61 |           1.26 |           1.61 |            4.05 |
| clu1k  | TourFold-v2(8) + süzülmüş omurga |           0.61 |           0.61 |           1.26 |           1.61 |            2.76 |
| tsplib | LKH-3 POPMUSIC (ascent yok)      |           4.66 |           7.91 |          13.98 |          29.28 |           51.58 |
| tsplib | LKH-3 varsayılan (α+ascent)      |         233.93 |         233.93 |         234.76 |         239.52 |          inf    |
| tsplib | Topluluk(8) + sayım+             |           7.98 |           7.98 |          17.48 |          51.1  |           74.6  |
| tsplib | Topluluk(8) omurgası sabit       |           7.8  |           8.79 |          17.48 |          54.95 |          inf    |
| tsplib | TourFold-v2(8)                   |           8.26 |           8.26 |          13.77 |          47.18 |           61.6  |
| tsplib | TourFold-v2(16)                  |          10.7  |          13.35 |          17.48 |          41.77 |           47.56 |
| tsplib | TourFold-v2(8) + süzülmüş omurga |           8.12 |           8.12 |          17.48 |          46.39 |           77.32 |
| uni10k | LKH-3 POPMUSIC (ascent yok)      |           8.74 |           8.74 |          11.16 |          18.36 |           37.47 |
| uni10k | LKH-3 varsayılan (α+ascent)      |         inf    |         inf    |         inf    |         inf    |          inf    |
| uni10k | Topluluk(8) + sayım+             |           9.65 |           9.65 |          25.41 |          34.69 |           63.91 |
| uni10k | Topluluk(8) omurgası sabit       |           9.65 |           9.65 |          25.35 |          29.67 |           55.59 |
| uni10k | TourFold-v2(8)                   |           9.65 |           9.65 |          25.3  |          31.69 |           61.95 |
| uni10k | TourFold-v2(16)                  |           9.65 |           9.65 |          42.25 |          52.65 |           73.47 |
| uni10k | TourFold-v2(8) + süzülmüş omurga |           9.65 |           9.65 |          25.3  |          30.84 |           59.16 |
| clu10k | LKH-3 POPMUSIC (ascent yok)      |          10.56 |          11.91 |          18.55 |          23.86 |           47.03 |
| clu10k | LKH-3 varsayılan (α+ascent)      |         inf    |         inf    |         inf    |         inf    |          inf    |
| clu10k | Topluluk(8) + sayım+             |          11.58 |          11.8  |          44.57 |          50.66 |           60.44 |
| clu10k | Topluluk(8) omurgası sabit       |          11.58 |          11.8  |          42.1  |          47.35 |           75.25 |
| clu10k | TourFold-v2(8)                   |          11.58 |          11.8  |          44.43 |          49.7  |           63.73 |
| clu10k | TourFold-v2(16)                  |          11.58 |          11.8  |          76.47 |          86.21 |           98.17 |
| clu10k | TourFold-v2(8) + süzülmüş omurga |          11.58 |          11.8  |          44.77 |          49.43 |           59.82 |
| uni20k | LKH-3 POPMUSIC (ascent yok)      |          20.11 |          20.11 |          23.44 |          58.72 |           80.59 |
| uni20k | LKH-3 varsayılan (α+ascent)      |         inf    |         inf    |         inf    |         inf    |          inf    |
| uni20k | Topluluk(8) + sayım+             |          23.92 |          23.92 |          42.17 |          88.97 |          119.38 |
| uni20k | Topluluk(8) omurgası sabit       |          23.92 |          23.92 |          42.17 |          91.9  |          140.51 |
| uni20k | TourFold-v2(8)                   |          23.92 |          23.92 |          42.17 |          87.18 |          175.2  |
| uni20k | TourFold-v2(16)                  |          23.92 |          23.92 |          42.17 |         152.75 |          213.74 |
| uni20k | TourFold-v2(8) + süzülmüş omurga |          23.92 |          23.92 |          42.17 |          93.14 |          170.68 |
| clu20k | LKH-3 POPMUSIC (ascent yok)      |          21.66 |          21.66 |          40.4  |          74.19 |          121.92 |
| clu20k | LKH-3 varsayılan (α+ascent)      |         inf    |         inf    |         inf    |         inf    |          inf    |
| clu20k | Topluluk(8) + sayım+             |          25.91 |          30.52 |         106.33 |         120.7  |          159.01 |
| clu20k | Topluluk(8) omurgası sabit       |          25.91 |          30.52 |         104.11 |         116.69 |          inf    |
| clu20k | TourFold-v2(8)                   |          25.91 |          30.52 |         105.95 |         124.42 |          156.68 |
| clu20k | TourFold-v2(16)                  |          25.91 |          30.52 |         185.9  |         205.62 |          248.4  |
| clu20k | TourFold-v2(8) + süzülmüş omurga |          25.91 |          30.52 |         105.94 |         119.25 |          152.16 |

Örnek bazında 'TourFold-v2(8)' kazanç/beraberlik/kayıp (tümü: v2/results/lkh_eval_v2_wtl.csv):

| set    | at        | LKH-3 POPMUSIC (ascent yok)   | LKH-3 varsayılan (α+ascent)   | Topluluk(8) + sayım+   | Topluluk(8) omurgası sabit   |
|:-------|:----------|:------------------------------|:------------------------------|:-----------------------|:-----------------------------|
| clu10k | gap@100%B | 3/0/5                         | 8/0/0                         | 4/0/4                  | 8/0/0                        |
| clu10k | gap@25%B  | 0/0/8                         | 8/0/0                         | 0/8/0                  | 0/8/0                        |
| clu1k  | gap@100%B | 5/2/1                         | 5/3/0                         | 2/6/0                  | 7/1/0                        |
| clu1k  | gap@25%B  | 2/0/6                         | 8/0/0                         | 0/8/0                  | 0/8/0                        |
| clu20k | gap@100%B | 1/0/3                         | 4/0/0                         | 3/0/1                  | 4/0/0                        |
| clu20k | gap@25%B  | 0/0/4                         | 4/0/0                         | 0/4/0                  | 0/4/0                        |
| tsplib | gap@100%B | 4/1/6                         | 9/0/2                         | 4/1/6                  | 10/0/1                       |
| tsplib | gap@25%B  | 6/0/5                         | 10/0/1                        | 9/1/1                  | 9/0/2                        |
| uni10k | gap@100%B | 4/0/4                         | 8/0/0                         | 6/0/2                  | 7/0/1                        |
| uni10k | gap@25%B  | 0/0/8                         | 8/0/0                         | 3/0/5                  | 1/0/7                        |
| uni1k  | gap@100%B | 4/4/0                         | 5/3/0                         | 2/5/1                  | 7/1/0                        |
| uni1k  | gap@25%B  | 4/1/3                         | 8/0/0                         | 5/2/1                  | 4/1/3                        |
| uni20k | gap@100%B | 0/0/4                         | 4/0/0                         | 0/0/4                  | 2/0/2                        |
| uni20k | gap@25%B  | 0/0/4                         | 4/0/0                         | 0/4/0                  | 0/4/0                        |

## Dosyalar

- v2/results: tablolar (csv, json) · v2/figures: grafikler · v2/runs: TourFold-v2 kontrol noktası · v2/log_v2.txt: ayrıntılı kayıt
