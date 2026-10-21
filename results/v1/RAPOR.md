# TourFold TSP — otomatik rapor

- Tarih: 2026-10-20 17:33
- Ortam: {'nproc': 12, 'ram_gb': 53.0, 'gpu': 'NVIDIA L4, 23034 MiB', 'numpy': '2.1.3'}
- Ölçek: L | topluluk M = 16 | kNN K = 16 | eğitim: 30000 örnek, n ∈ [100, 500], 12 epoch
- Aşama süreleri (dk, tüm oturumların toplamı): {'1-lkh': 0.1, '1-selftest': 0.0, '2-profile': 0.0, '3-eval_sets': 0.1, '3-eval_ref': 326.4, '3-classic_coverage': 2.2, '4-train_data': 778.4, '5-train_tourfold': 108.0, '5-train_single': 67.2, '6-decision': 0.9, '7-lkh_eval': 47.8}
- TSPLIB: 31 örnek

## 1. Öz-sınama

- OK: LKH geçerli bir tur üretti
- OK: tur uzunluğu LKH'nin bildirdiğiyle aynı
- OK: LKH iyileşme izi okundu
- OK: LKH ön-işleme süresi okundu
- OK: LKH aday (CANDIDATE_FILE) ve ceza (PI_FILE) dosyaları okundu
- OK: kendi yazdığımız aday/π dosyaları LKH'ye kabul edildi
- OK: tur topluluğu üretildi (POPMUSIC adaylar, ascent yok)
- OK: başlangıç turu + dışarıdan aday dosyasıyla LKH çalıştı
- OK: π birimi (PRECISION = 100) doğrulandı
- OK: sabit kenarlar (FIXED_EDGES_SECTION) LKH turunda korundu

## 2. LKH ön-işleme süresi (sn)

|     n |   POPMUSIC, no ascent |   alpha+ascent |   alpha, no ascent |
|------:|----------------------:|---------------:|-------------------:|
|  1000 |                  0.42 |           1.13 |               0.05 |
|  2000 |                  1.13 |           5.77 |               0.19 |
|  5000 |                  3.65 |          36.66 |               2.28 |
| 10000 |                  8.17 |         161.71 |              11.06 |
| 20000 |                 17.21 |         710.97 |              45.3  |

## 3. Eğitim (doğrulama; eksik tur kenarı / 1000 düğüm, ilk 5 aday)

| model           |   epoch |   son_kayip |   eksik_1000_model |   eksik_1000_alpha_ascent |   en_iyi_eksik_1000 |
|:----------------|--------:|------------:|-------------------:|--------------------------:|--------------------:|
| TourFold        |      12 |       1.615 |              7.234 |                      4.56 |               7.234 |
| single-instance |      12 |       2.265 |              7.807 |                      4.56 |               7.804 |

## 4. Karar deneyi

Yöntemler: TourFold = tur topluluğunu okuyan model · single-instance = aynı model, topluluk girdisi yok (ablasyon) · count = topluluktaki kullanım sıklığı (sayım) · alpha+ascent = LKH-3 varsayılan adayları · POPMUSIC = LKH'nin ascent'siz adayları. Gruplar: val = eğitim dağılımından doğrulama; uni/clu = düzgün/kümeli rastgele örnekler; tsplib = TSPLIB örnekleri. Not: referans turlar LKH ile α-8 aday grafiğinde bulunduğu için alpha+ascent sütunu biraz iyimserdir.

Referans turun kenarlarından, düğümün ilk 5 adayı dışında kalanlar (1000 düğüm başına; küçük = iyi):

| method          |   clu10k |   clu1k |   tsplib |   uni10k |   uni1k |   val |
|:----------------|---------:|--------:|---------:|---------:|--------:|------:|
| POPMUSIC        |     6.18 |    5.62 |     7.25 |     5.6  |    5.23 |  3.95 |
| TourFold        |     0.82 |    4.56 |     4.1  |     0.34 |    0.25 |  7.23 |
| alpha+ascent    |     2.46 |    3.89 |     4.69 |     2.79 |    2.27 |  4.56 |
| count           |     1.84 |    5    |     4.39 |     1.12 |    0.72 |  7.6  |
| single-instance |     0.89 |    5.09 |     4.12 |     0.44 |    0.44 |  7.81 |

İlk 2 aday içinde kalan tur kenarı oranı (büyük = iyi):

| method          |   clu10k |   clu1k |   tsplib |   uni10k |   uni1k |    val |
|:----------------|---------:|--------:|---------:|---------:|--------:|-------:|
| POPMUSIC        |   0.851  |  0.8451 |   0.8628 |   0.8556 |  0.8533 | 0.85   |
| TourFold        |   0.9216 |  0.9388 |   0.9173 |   0.9339 |  0.9467 | 0.9624 |
| alpha+ascent    |   0.8948 |  0.8845 |   0.9015 |   0.8964 |  0.8933 | 0.8866 |
| count           |   0.9281 |  0.9379 |   0.9179 |   0.9376 |  0.9451 | 0.9518 |
| single-instance |   0.9012 |  0.8877 |   0.8919 |   0.9076 |  0.9035 | 0.8823 |

Klasik aday kümeleri, aynı referans turlar (eksik / 1000):

| method            |   clu10k |   clu1k |   tsplib |   uni10k |   uni1k |
|:------------------|---------:|--------:|---------:|---------:|--------:|
| alpha, no ascent  |    22.24 |   27.02 |    18.23 |     15.2 |   16.95 |
| nearest neighbour |    18.23 |   23.41 |    21.06 |     17.5 |   19.52 |

Kenar sabitleme: doğrulamada kesinlik ≥ 0.999 veren en küçük eşik τ = 0.99 (kesinlik ve duyarlılık kenar düzeyinde, tüm örnekler üzerinden toplanarak):

| group   | method                     |   precision |   recall |   fixed_frac |
|:--------|:---------------------------|------------:|---------:|-------------:|
| clu10k  | TourFold                   |      1      |   0      |       0      |
| clu10k  | count backbone (all tours) |      0.9967 |   0.6241 |       0.6262 |
| clu10k  | single-instance            |    nan      |   0      |       0      |
| clu1k   | TourFold                   |      1      |   0.0029 |       0.0029 |
| clu1k   | count backbone (all tours) |      0.9972 |   0.683  |       0.685  |
| clu1k   | single-instance            |    nan      |   0      |       0      |
| tsplib  | TourFold                   |      0.9991 |   0.0083 |       0.0083 |
| tsplib  | count backbone (all tours) |      0.9971 |   0.6397 |       0.6416 |
| tsplib  | single-instance            |    nan      |   0      |       0      |
| uni10k  | TourFold                   |      1      |   0.0012 |       0.0012 |
| uni10k  | count backbone (all tours) |      0.9977 |   0.6433 |       0.6448 |
| uni10k  | single-instance            |    nan      |   0      |       0      |
| uni1k   | TourFold                   |      1      |   0.0081 |       0.0081 |
| uni1k   | count backbone (all tours) |      0.9982 |   0.6832 |       0.6844 |
| uni1k   | single-instance            |    nan      |   0      |       0      |
| val     | TourFold                   |      0.999  |   0.0887 |       0.0888 |
| val     | count backbone (all tours) |      0.9981 |   0.7677 |       0.7691 |
| val     | single-instance            |    nan      |   0      |       0      |

### Hüküm

- **clu10k**: TourFold 0.83 · sayım 1.84 · POPMUSIC 6.17 · α+ascent 2.46 · tek-örnek 0.89 → TourFold, sayım ve POPMUSIC'ten daha iyi (EVET) · örnek bazında kazanç/beraberlik/kayıp: sayıma 8/0/0, POPMUSIC'e 8/0/0, α+ascent'e 8/0/0, tek-örneğe 3/4/1
- **clu1k**: TourFold 4.56 · sayım 5.00 · POPMUSIC 5.62 · α+ascent 3.89 · tek-örnek 5.09 → TourFold, sayım ve POPMUSIC'ten daha iyi (EVET) · örnek bazında kazanç/beraberlik/kayıp: sayıma 19/43/2, POPMUSIC'e 39/5/20, α+ascent'e 19/13/32, tek-örneğe 29/30/5
- **tsplib**: TourFold 4.10 · sayım 4.39 · POPMUSIC 7.25 · α+ascent 4.69 · tek-örnek 4.12 → TourFold, sayım ve POPMUSIC'ten daha iyi (EVET) · örnek bazında kazanç/beraberlik/kayıp: sayıma 18/11/2, POPMUSIC'e 28/0/3, α+ascent'e 20/4/7, tek-örneğe 14/10/7
- **uni10k**: TourFold 0.34 · sayım 1.12 · POPMUSIC 5.60 · α+ascent 2.79 · tek-örnek 0.44 → TourFold, sayım ve POPMUSIC'ten daha iyi (EVET) · örnek bazında kazanç/beraberlik/kayıp: sayıma 8/0/0, POPMUSIC'e 8/0/0, α+ascent'e 8/0/0, tek-örneğe 6/0/2
- **uni1k**: TourFold 0.25 · sayım 0.72 · POPMUSIC 5.23 · α+ascent 2.27 · tek-örnek 0.44 → TourFold, sayım ve POPMUSIC'ten daha iyi (EVET) · örnek bazında kazanç/beraberlik/kayıp: sayıma 24/39/1, POPMUSIC'e 63/1/0, α+ascent'e 53/10/1, tek-örneğe 15/41/8
- **val**: TourFold 7.23 · sayım 7.60 · POPMUSIC 3.95 · α+ascent 4.56 · tek-örnek 7.81 → TourFold, sayım ve POPMUSIC'ten daha iyi değil (HAYIR) · örnek bazında kazanç/beraberlik/kayıp: sayıma 83/910/7, POPMUSIC'e 381/240/379, α+ascent'e 328/308/364, tek-örneğe 170/800/30

## 5. LKH ile süre–kalite (aynı oturum; toplam süre = topluluk + model + LKH)

Rehberli koşularda başlangıç turu topluluğun en iyi turudur; LKH bu turun kenarlarını aday kümesine kendisi ekler (tüm rehberli yapılandırmalar için aynı). Sabit kenarlı koşularda LKH sabit kenarları 0 uzunlukla saydığından bildirdiği maliyetler gerçek tur uzunluğuna çevrildi.

Bütçenin %10, %25, %50 ve %100'ünde en iyi bilinene göre fark (%, ortalama; küçük = iyi; nan = o ana kadar hiç tur yok, ör. LKH-3 varsayılan ascent'i bitirmeden):

| set    | config                             |   gap@10%B |   gap@25%B |   gap@50%B |   gap@100%B |   turlu@10%B |   turlu@25%B |   örnek |
|:-------|:-----------------------------------|-----------:|-----------:|-----------:|------------:|-------------:|-------------:|--------:|
| uni1k  | LKH-3 varsayılan (α+ascent)        |     0.0131 |     0.0076 |     0.0062 |      0.0062 |            8 |            8 |       8 |
| uni1k  | LKH-3 POPMUSIC (ascent yok)        |     0.0116 |     0.0098 |     0.0037 |      0.0029 |            8 |            8 |       8 |
| uni1k  | Topluluk + sayım adayları          |     0.006  |     0.0028 |     0.0026 |      0.0026 |            8 |            8 |       8 |
| uni1k  | Tek-örnek model adayları           |     0.0055 |     0.0013 |     0      |      0      |            8 |            8 |       8 |
| uni1k  | TourFold adayları                  |     0.0035 |     0.0019 |     0.0004 |      0.0004 |            8 |            8 |       8 |
| uni1k  | TourFold adayları + π              |     0.0036 |     0.0013 |     0.0013 |      0      |            8 |            8 |       8 |
| uni1k  | TourFold adayları + sabit kenarlar |     0.0061 |     0.0019 |     0.0019 |      0.0019 |            8 |            8 |       8 |
| clu1k  | LKH-3 varsayılan (α+ascent)        |     0.4803 |     0.2753 |     0.0735 |      0.011  |            8 |            8 |       8 |
| clu1k  | LKH-3 POPMUSIC (ascent yok)        |     0.0186 |     0.0139 |     0.0081 |      0.0004 |            8 |            8 |       8 |
| clu1k  | Topluluk + sayım adayları          |     0.1101 |     0.0236 |     0.0153 |      0.0101 |            8 |            8 |       8 |
| clu1k  | Tek-örnek model adayları           |     0.1101 |     0.0158 |     0.0138 |      0.0138 |            8 |            8 |       8 |
| clu1k  | TourFold adayları                  |     0.1101 |     0.0142 |     0.0138 |      0.0057 |            8 |            8 |       8 |
| clu1k  | TourFold adayları + π              |     0.1101 |     0.019  |     0.0138 |      0.0138 |            8 |            8 |       8 |
| clu1k  | TourFold adayları + sabit kenarlar |     0.1101 |     0.0142 |     0.0138 |      0.0057 |            8 |            8 |       8 |
| uni10k | LKH-3 varsayılan (α+ascent)        |   nan      |   nan      |   nan      |      0.0036 |            0 |            0 |       2 |
| uni10k | LKH-3 POPMUSIC (ascent yok)        |     0.0664 |     0.0493 |     0.0298 |      0.0225 |            2 |            2 |       2 |
| uni10k | Topluluk + sayım adayları          |     0.2125 |     0.0457 |     0.035  |      0.023  |            2 |            2 |       2 |
| uni10k | Tek-örnek model adayları           |     0.2125 |     0.0589 |     0.0345 |      0.0224 |            2 |            2 |       2 |
| uni10k | TourFold adayları                  |     0.2125 |     0.0615 |     0.0265 |      0.0146 |            2 |            2 |       2 |
| uni10k | TourFold adayları + π              |     0.2125 |     0.057  |     0.0376 |      0.0305 |            2 |            2 |       2 |
| uni10k | TourFold adayları + sabit kenarlar |     0.2125 |     0.0576 |     0.0251 |      0.0206 |            2 |            2 |       2 |
| clu10k | LKH-3 varsayılan (α+ascent)        |   nan      |   nan      |   nan      |      0.0094 |            0 |            0 |       2 |
| clu10k | LKH-3 POPMUSIC (ascent yok)        |     0.0876 |     0.0339 |     0.0265 |      0.0143 |            2 |            2 |       2 |
| clu10k | Topluluk + sayım adayları          |     0.3512 |     0.2488 |     0.03   |      0.0114 |            2 |            2 |       2 |
| clu10k | Tek-örnek model adayları           |     0.3512 |     0.3077 |     0.0183 |      0.0047 |            2 |            2 |       2 |
| clu10k | TourFold adayları                  |     0.3512 |     0.3077 |     0.0257 |      0.0183 |            2 |            2 |       2 |
| clu10k | TourFold adayları + π              |     0.3512 |     0.3077 |     0.0187 |      0.007  |            2 |            2 |       2 |
| clu10k | TourFold adayları + sabit kenarlar |     0.3512 |     0.3077 |     0.0257 |      0.0183 |            2 |            2 |       2 |
| tsplib | LKH-3 varsayılan (α+ascent)        |     0.7406 |     0.0479 |     0.0432 |      0.0309 |           15 |           20 |      20 |
| tsplib | LKH-3 POPMUSIC (ascent yok)        |     0.1406 |     0.0854 |     0.0337 |      0.0297 |           20 |           20 |      20 |
| tsplib | Topluluk + sayım adayları          |     0.1236 |     0.029  |     0.0193 |      0.0104 |           20 |           20 |      20 |
| tsplib | Tek-örnek model adayları           |     0.1152 |     0.0402 |     0.0289 |      0.0244 |           20 |           20 |      20 |
| tsplib | TourFold adayları                  |     0.1205 |     0.0377 |     0.0317 |      0.0165 |           20 |           20 |      20 |
| tsplib | TourFold adayları + π              |     0.1259 |     0.0339 |     0.0248 |      0.0189 |           20 |           20 |      20 |
| tsplib | TourFold adayları + sabit kenarlar |     0.122  |     0.04   |     0.037  |      0.0228 |           20 |           20 |      20 |

Hedef farka ulaşma süresi (sn, medyan; inf = bütçe içinde ulaşılamadı):

| set    | config                             |   time_to_1.0% |   time_to_0.5% |   time_to_0.2% |   time_to_0.1% |
|:-------|:-----------------------------------|---------------:|---------------:|---------------:|---------------:|
| uni1k  | LKH-3 varsayılan (α+ascent)        |           1.23 |           1.23 |           1.25 |           1.56 |
| uni1k  | LKH-3 POPMUSIC (ascent yok)        |           0.49 |           0.49 |           0.52 |           0.56 |
| uni1k  | Topluluk + sayım adayları          |           0.49 |           0.49 |           0.59 |           0.97 |
| uni1k  | Tek-örnek model adayları           |           0.49 |           0.49 |           0.59 |           0.97 |
| uni1k  | TourFold adayları                  |           0.49 |           0.49 |           0.59 |           0.97 |
| uni1k  | TourFold adayları + π              |           0.49 |           0.49 |           0.59 |           0.97 |
| uni1k  | TourFold adayları + sabit kenarlar |           0.49 |           0.49 |           0.59 |           0.97 |
| clu1k  | LKH-3 varsayılan (α+ascent)        |           1.74 |           2.33 |           2.44 |           2.44 |
| clu1k  | LKH-3 POPMUSIC (ascent yok)        |           0.65 |           0.7  |           0.8  |           1.04 |
| clu1k  | Topluluk + sayım adayları          |           0.63 |           0.63 |           1.28 |           1.62 |
| clu1k  | Tek-örnek model adayları           |           0.63 |           0.63 |           1.28 |           1.62 |
| clu1k  | TourFold adayları                  |           0.63 |           0.63 |           1.28 |           1.62 |
| clu1k  | TourFold adayları + π              |           0.63 |           0.63 |           1.28 |           1.62 |
| clu1k  | TourFold adayları + sabit kenarlar |           0.63 |           0.63 |           1.28 |           1.62 |
| uni10k | LKH-3 varsayılan (α+ascent)        |         154.99 |         154.99 |         154.99 |         157.26 |
| uni10k | LKH-3 POPMUSIC (ascent yok)        |           9    |           9    |          11.16 |          18.01 |
| uni10k | Topluluk + sayım adayları          |           9.34 |           9.34 |          34.75 |          49.94 |
| uni10k | Tek-örnek model adayları           |           9.34 |           9.34 |          34.81 |          53.02 |
| uni10k | TourFold adayları                  |           9.34 |           9.34 |          34.91 |          51.21 |
| uni10k | TourFold adayları + π              |           9.34 |           9.34 |          34.91 |          47.82 |
| uni10k | TourFold adayları + sabit kenarlar |           9.34 |           9.34 |          34.91 |          50.43 |
| clu10k | LKH-3 varsayılan (α+ascent)        |         157.26 |         158.17 |         167.91 |         177.42 |
| clu10k | LKH-3 POPMUSIC (ascent yok)        |          11.61 |          14.74 |          22.29 |          25.62 |
| clu10k | Topluluk + sayım adayları          |          11.32 |          13.34 |          78.05 |          88.92 |
| clu10k | Tek-örnek model adayları           |          11.32 |          13.34 |          78.57 |          87.43 |
| clu10k | TourFold adayları                  |          11.32 |          13.34 |          79.68 |          86.9  |
| clu10k | TourFold adayları + π              |          11.32 |          13.34 |          76.91 |          80.89 |
| clu10k | TourFold adayları + sabit kenarlar |          11.32 |          13.34 |          79.73 |          86.99 |
| tsplib | LKH-3 varsayılan (α+ascent)        |           2.58 |           2.58 |           3.38 |           3.73 |
| tsplib | LKH-3 POPMUSIC (ascent yok)        |           0.75 |           0.82 |           1.13 |           1.76 |
| tsplib | Topluluk + sayım adayları          |           0.76 |           0.82 |           1.34 |           2.6  |
| tsplib | Tek-örnek model adayları           |           0.76 |           0.82 |           1.34 |           2.69 |
| tsplib | TourFold adayları                  |           0.76 |           0.82 |           1.34 |           2.79 |
| tsplib | TourFold adayları + π              |           0.76 |           0.82 |           1.34 |           2.88 |
| tsplib | TourFold adayları + sabit kenarlar |           0.76 |           0.82 |           1.34 |           2.72 |

Örnek bazında 'TourFold adayları' kazanç/beraberlik/kayıp (tümü: results/lkh_eval_wtl.csv):

| set    | at        | LKH-3 POPMUSIC (ascent yok)   | LKH-3 varsayılan (α+ascent)   | Tek-örnek model adayları   | Topluluk + sayım adayları   |
|:-------|:----------|:------------------------------|:------------------------------|:---------------------------|:----------------------------|
| clu10k | gap@100%B | 0/0/2                         | 0/0/2                         | 0/0/2                      | 0/0/2                       |
| clu10k | gap@25%B  | 0/0/2                         | 2/0/0                         | 0/2/0                      | 0/1/1                       |
| clu1k  | gap@100%B | 1/6/1                         | 1/6/1                         | 2/6/0                      | 2/6/0                       |
| clu1k  | gap@25%B  | 2/4/2                         | 4/3/1                         | 1/7/0                      | 3/5/0                       |
| tsplib | gap@100%B | 5/13/2                        | 6/11/3                        | 3/13/4                     | 2/17/1                      |
| tsplib | gap@25%B  | 6/9/5                         | 8/6/6                         | 3/10/7                     | 6/10/4                      |
| uni10k | gap@100%B | 2/0/0                         | 0/0/2                         | 2/0/0                      | 2/0/0                       |
| uni10k | gap@25%B  | 1/0/1                         | 2/0/0                         | 1/0/1                      | 0/0/2                       |
| uni1k  | gap@100%B | 3/4/1                         | 4/4/0                         | 0/7/1                      | 2/6/0                       |
| uni1k  | gap@25%B  | 4/3/1                         | 4/4/0                         | 1/6/1                      | 1/7/0                       |

## Dosyalar

- results/: tablolar (csv, json) · figures/: grafikler · runs/: model kontrol noktaları · log.txt: ayrıntılı kayıt
