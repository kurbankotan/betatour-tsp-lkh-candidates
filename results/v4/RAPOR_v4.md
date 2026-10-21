# TourFold TSP v4 — çok çekirdekli karşılaştırma (otomatik rapor)

- Tarih: 2026-10-21 10:22
- Ortam: {'nproc': 12, 'ram_gb': 53.0, 'gpu': 'NVIDIA L4, 23034 MiB', 'numpy': '2.1.3'}
- Örnek başına 6 paralel LKH koşusu (tohumlar [101, 102, 103, 104, 105, 106]); TourFold listesi: modelin ilk 3 adayı + koşunun kendi POPMUSIC listesi (5 aday)
- Adım süreleri (dk, tüm oturumların toplamı): {'1-LKH': 0.1, '1-öz-sınama': 0.0, '2-örnekler': 2.7, '2-model': 0.1, '2-model denetimi': 0.2, '3-çok çekirdekli LKH': 218.9, '4-analiz': 0.3}

## 1. Düzen

- **LKH-3 POPMUSIC ×6**: 6 bağımsız LKH-3 POPMUSIC koşusu (ascent yok), her biri bir çekirdekte, bütçenin tamamı boyunca.
- **TourFold ×6**: aynı 6 koşu ilk denemeden sonra durur (aynı tohum, aynı ilk deneme). TourFold-v2 bu 6 turu topluluk olarak okur. Her koşu kendi turundan, TourFold'un ilk 3 adayı + kendi POPMUSIC listesiyle devam eder. En yavaş ilk denemeyi bekleme, model ve liste süresi bütçeden düşülür.
- Her an 6 koşunun en iyi turu sayılır. En iyi bilinen: TSPLIB'de optimum; diğerlerinde referans tur ile v2, v3 ve v4 koşularının en iyisi. Örnek bazında iki yapılandırmanın sırası değişir.

## 2. Öz-sınama ve model denetimi

Öz-sınama: tüm kontroller geçti.

TourFold-v2, doğrulama örneklerinde eksik tur kenarı / 1000 düğüm (ilk 5 aday): 4 tur: 0.91, 6 tur: 0.71, 8 tur: 0.56

## 3. Sonuçlar (51/51 örnek)

6 koşunun en iyisinin, bütçenin %10, %25, %50 ve %100'ünde en iyi bilinene göre farkı (%, ortalama; nan = o ana kadar tur yok):

| set    | config            |   gap@10%B |   gap@25%B |   gap@50%B |   gap@100%B |   turlu@10%B |   turlu@25%B |   örnek |
|:-------|:------------------|-----------:|-----------:|-----------:|------------:|-------------:|-------------:|--------:|
| clu1k  | LKH-3 POPMUSIC ×6 |   nan      |     0.0119 |     0.0062 |      0.0008 |            0 |            8 |       8 |
| clu1k  | TourFold ×6       |   nan      |     0.0228 |     0.0109 |      0      |            0 |            8 |       8 |
| uni1k  | LKH-3 POPMUSIC ×6 |     0.0682 |     0.0096 |     0.0023 |      0.0015 |            8 |            8 |       8 |
| uni1k  | TourFold ×6       |     0.0869 |     0.0044 |     0.0006 |      0      |            8 |            8 |       8 |
| tsplib | LKH-3 POPMUSIC ×6 |     0.0861 |     0.0329 |     0.0171 |      0.0104 |           11 |           11 |      11 |
| tsplib | TourFold ×6       |     0.065  |     0.0232 |     0.0148 |      0.0107 |           11 |           11 |      11 |
| clu10k | LKH-3 POPMUSIC ×6 |     0.2904 |     0.0541 |     0.0217 |      0.0088 |            8 |            8 |       8 |
| clu10k | TourFold ×6       |     0.3305 |     0.0562 |     0.0235 |      0.0096 |            8 |            8 |       8 |
| uni10k | LKH-3 POPMUSIC ×6 |     0.1383 |     0.0463 |     0.0229 |      0.0142 |            8 |            8 |       8 |
| uni10k | TourFold ×6       |     0.1752 |     0.0467 |     0.018  |      0.0084 |            8 |            8 |       8 |
| clu20k | LKH-3 POPMUSIC ×6 |     0.2071 |     0.0607 |     0.0173 |      0.0007 |            4 |            4 |       4 |
| clu20k | TourFold ×6       |     0.3491 |     0.0838 |     0.0384 |      0.0095 |            4 |            4 |       4 |
| uni20k | LKH-3 POPMUSIC ×6 |     0.1353 |     0.0426 |     0.0182 |      0.0007 |            4 |            4 |       4 |
| uni20k | TourFold ×6       |     0.1521 |     0.063  |     0.0404 |      0.0212 |            4 |            4 |       4 |

Hedef farka ulaşma süresi (sn, medyan; inf = bütçe içinde ulaşılamadı):

| set    | config            |   time_to_1.0% |   time_to_0.5% |   time_to_0.2% |   time_to_0.1% |   time_to_0.05% |
|:-------|:------------------|---------------:|---------------:|---------------:|---------------:|----------------:|
| clu1k  | LKH-3 POPMUSIC ×6 |           0.62 |           0.62 |           0.66 |           0.66 |            0.67 |
| clu1k  | TourFold ×6       |           0.62 |           0.62 |           0.69 |           0.71 |            0.71 |
| uni1k  | LKH-3 POPMUSIC ×6 |           0.46 |           0.46 |           0.47 |           0.49 |            0.5  |
| uni1k  | TourFold ×6       |           0.46 |           0.46 |           0.47 |           0.48 |            0.63 |
| tsplib | LKH-3 POPMUSIC ×6 |           4.78 |           4.78 |          11.45 |          20.3  |           38.62 |
| tsplib | TourFold ×6       |           4.76 |           4.76 |           7.94 |           9.63 |           14.53 |
| clu10k | LKH-3 POPMUSIC ×6 |          10.46 |          10.46 |          13.09 |          18.99 |           28.53 |
| clu10k | TourFold ×6       |          10.49 |          10.49 |          17.02 |          20.95 |           30.51 |
| uni10k | LKH-3 POPMUSIC ×6 |           8.68 |           8.68 |           9.77 |          14.5  |           31.29 |
| uni10k | TourFold ×6       |           8.73 |           8.73 |          11.43 |          16.23 |           27.42 |
| clu20k | LKH-3 POPMUSIC ×6 |          21.07 |          21.21 |          29.99 |          45.88 |           86.45 |
| clu20k | TourFold ×6       |          21.17 |          21.24 |          34.88 |          63.06 |          121.17 |
| uni20k | LKH-3 POPMUSIC ×6 |          19.38 |          19.38 |          21.5  |          37.85 |           64.93 |
| uni20k | TourFold ×6       |          19.49 |          19.49 |          27.67 |          40.85 |          112.21 |

### TourFold ×6 ile LKH-3 POPMUSIC ×6

Örnek bazında kazanç/beraberlik/kayıp (K/B/K = TourFold iyi / eşit / kötü), boyut sınıflarına göre ve tümü; p: iki yönlü işaret testi (beraberlikler hariç) ve Wilcoxon işaretli sıra testi; ortalama fark yüzde puan (eksi = TourFold iyi):

| bütçe   | 1k     | 10k   | TSPLIB   | 20k   | tümü     |   işaret testi p |   Wilcoxon p |   ort. fark (puan) |
|:--------|:-------|:------|:---------|:------|:---------|-----------------:|-------------:|-------------------:|
| %25     | 4/7/5  | 8/0/8 | 5/1/5    | 1/0/7 | 18/8/25  |            0.36  |        0.131 |             0.0026 |
| %50     | 3/11/2 | 8/0/8 | 4/2/5    | 0/0/8 | 15/13/23 |            0.256 |        0.455 |             0.0029 |
| %100    | 2/13/1 | 8/0/8 | 6/2/3    | 2/0/6 | 18/15/18 |            1     |        0.969 |             0.0012 |

Rakibin bütçe sonundaki kalitesine ulaşma süresi (bütçenin yüzdesi, medyan; 999 = bütçe içinde ulaşamadı) ve bütçe içinde ulaşılan örneklerin oranı:

| sınıf   |   örnek |   TourFold → tabanın son kalitesi: medyan süre (bütçe %) |   TourFold bütçe içinde ulaştı (%) |   taban → TourFold'un son kalitesi: medyan süre (bütçe %) |   taban bütçe içinde ulaştı (%) |
|:--------|--------:|---------------------------------------------------------:|-----------------------------------:|----------------------------------------------------------:|--------------------------------:|
| 1k      |      16 |                                                     28.2 |                               93.8 |                                                      25.2 |                            87.5 |
| 10k     |      16 |                                                     98.3 |                               50   |                                                     548.4 |                            50   |
| TSPLIB  |      11 |                                                     50.7 |                               72.7 |                                                     999   |                            45.5 |
| 20k     |       8 |                                                    999   |                               25   |                                                      61.6 |                            75   |
| tümü    |      51 |                                                     59.9 |                               64.7 |                                                      80.8 |                            64.7 |

Koşu bazında (aynı örnek, aynı tohum, ilk deneme ortak; her koşu tek başına):

| bütçe   |   koşu çifti | K/B/K      |   işaret testi p |   Wilcoxon p |
|:--------|-------------:|:-----------|-----------------:|-------------:|
| %50     |          306 | 142/43/121 |            0.217 |       0.0976 |
| %100    |          306 | 134/64/108 |            0.108 |       0.0742 |

İlk denemesi iki yapılandırmada birebir aynı olan koşu oranı: 1.000

### TourFold'un ek süresi (sn, ortalama; bütçeden düşülür)

| set    |   ilk_denemeler |   model |   liste |   devam_anı |   bütçe_payı_yüzde |   taban_başlama_gecikmesi |   tf_ilk3_popmusicte |
|:-------|----------------:|--------:|--------:|------------:|-------------------:|--------------------------:|---------------------:|
| clu1k  |           0.782 |   0.062 |   0.013 |       0.872 |             17.444 |                     0.002 |                0.953 |
| uni1k  |           0.521 |   0.062 |   0.013 |       0.613 |             12.25  |                     0.002 |                0.96  |
| tsplib |           9.458 |   0.306 |   0.11  |       9.981 |              4.761 |                     0.002 |                0.936 |
| clu10k |          13.662 |   0.323 |   0.121 |      14.224 |             11.853 |                     0.01  |                0.96  |
| uni10k |          10.311 |   0.32  |   0.117 |      10.865 |              9.054 |                     0.01  |                0.963 |
| clu20k |          29.641 |   0.7   |   0.232 |      30.807 |             10.269 |                     0.017 |                0.961 |
| uni20k |          25.559 |   0.701 |   0.236 |      26.727 |              8.909 |                     0.017 |                0.963 |

ilk_denemeler: en yavaş ilk denemenin bitişi · model, liste: TourFold-v2'nin tahmini ve listelerin kurulması · devam_anı: koşuların TourFold adaylarıyla yeniden başladığı an · taban_başlama_gecikmesi: taban koşularının başlama gecikmesi · tf_ilk3_popmusicte: TourFold'un ilk 3 adayının koşunun kendi POPMUSIC listesinde de bulunma oranı.

### Bağlam: tek koşu (v3) ve paralel koşular, bütçenin tamamında fark (%, ortalama)

| set    |   tek koşu, POPMUSIC (v3) |   6 koşunun en iyisi, POPMUSIC |   6 koşunun en iyisi, TourFold |
|:-------|--------------------------:|-------------------------------:|-------------------------------:|
| uni1k  |                    0.0098 |                         0.0015 |                         0      |
| clu1k  |                    0.0155 |                         0.0008 |                         0      |
| tsplib |                    0.0209 |                         0.0104 |                         0.0107 |
| uni10k |                    0.0291 |                         0.0142 |                         0.0084 |
| clu10k |                    0.0252 |                         0.0088 |                         0.0096 |
| uni20k |                    0.0248 |                         0.0007 |                         0.0212 |
| clu20k |                    0.0243 |                         0.0007 |                         0.0095 |

## Dosyalar

- v4/results: tablolar (csv, json); instances/: örnek başına tüm koşuların izleri · v4/figures: grafikler · v4/log_v4.txt: ayrıntılı kayıt
