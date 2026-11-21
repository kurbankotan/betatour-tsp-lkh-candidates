# BetaTour run v27 (pilot): birleşim grafında tam çözüm

Run v26'da altı LKH-3 sürecinin turlarını bütçenin yarısında birleştirmek, süreçleri bütçe sonuna kadar çalıştırmaktan
daha kısa turlar verdi; GA-EAX ile LKH-3'ün kendi tur birleştirmesi arasındaki fark anlamlı çıkmadı. Bu pilot iki soruyu
sorar:

1. **Matematik:** altı sürecin en iyi turlarının **birleşim grafındaki en kısa tur** (SCIP ile dal-ve-kesi, kanıtlanmış
   en iyi) GA-EAX'in bulduğu turdan daha kısa mı?
2. **Derin öğrenme:** birleşim grafına BetaTour'un en olası yeni kenarlarını (n'nin %2'si) eklemek, aynı sayıda en kısa
   yeni kenar eklemekten ve hiç eklememekten daha kısa tur veriyor mu?

Kollar (her örnekte tek bir kayıtlı koşudan; kesme anı bütçenin yarısı): `lkh6` (altı süreç bütçe sonuna kadar),
`trials_50` ve `best6_50` (GA-EAX; 102 tur ve altı en iyi tur), `dual_50` (ikisinden kısa olanı; run v26'nın birincil
kolu), `lkhmerge_50` (LKH-3'ün kendi tur birleştirmesi), `exact6_50` (birleşimde tam çözüm), `exact6beta_50`
(birleşim + BetaTour kenarları), `exact6near_50` (birleşim + en kısa kenarlar). Örnekler: 12 yeni 10.000 düğümlük
örnek (6 düzgün, 6 kümeli; 120 s) ve 2 yeni 20.000 düğümlük örnek (240 s; betimsel).

Okumalar ve karar kuralı `experiments/PROTOCOL_v27.md` dosyasındadır; defter, protokol ve defterin kendisi GitHub'da
(main) kurulduğu biçimde değilse başlamaz. Önce depoyu commit ve push edin.

**Çalıştırma:** Çalışma zamanı türü **L4 GPU + Yüksek RAM** (12 vCPU, 53 GB) seçin ve bütün hücreleri sırayla
çalıştırın. Tahmini süre: kurulum (PySCIPOpt kurulumu, LKH-3 ve GA-EAX derlemeleri, model) 6-8 dk; öz sınama 2-3 dk;
10.000 düğümlük 12 örnek, her biri yaklaşık 5 dk, 1 saat; 20.000 düğümlük 2 örnek, her biri yaklaşık 10 dk, 20 dk;
çözümleme ve arşiv 1 dk; toplam yaklaşık 1,5 saat. Biten örnekler Drive'a yazılır
(`MyDrive/BetaTour/v27_union14`) ve defter yeniden çalıştırılınca atlanır; makine değişirse yarım kalan örnek baştan
koşar. İlk hücredeki `MAX_NEW_INSTANCES` deneyi oturumlara böler. Sonunda indirilen
`v27_union14_results.zip` dosyasını gönderin.
