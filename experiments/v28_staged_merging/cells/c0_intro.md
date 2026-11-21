# BetaTour run v28 (pilot): kademeli birleştirme

Run v26'da 10.000 düğümde GA-EAX'e 102 tur (her sürecin en iyi turu ve son 16 deneme turu) vermek, yalnızca altı en
iyi turu vermekten daha kısa turlar verdi; 20.000 ve 50.000 düğümde ise GA-EAX 102 turu kalan sürede yakınsatamadı.
Run v27'de birleşim grafında tam çözüm ve BetaTour kenarları GA-EAX birleştirmesini geçemedi. Bu pilot şunu sorar:
102 tur iki aşamada birleştirilirse (önce altı grupta aynı anda, sonra grupların sonuç turları ve altı en iyi tur), 20.000
düğümde iki GA-EAX sürecinden (run v26'nın birincil kolu) ve LKH-3'ün kendi tur birleştirmesinden daha kısa tur verir mi?

Kollar (her örnekte tek bir kayıtlı koşudan; kesme anı bütçenin yarısı): `lkh6` (altı süreç bütçe sonuna kadar),
`trials_50` ve `best6_50` (GA-EAX; 102 tur ve altı en iyi tur), `dual_50` (ikisinden kısa olanı; run v26'nın birincil
kolu), `lkhmerge_50` (LKH-3'ün kendi tur birleştirmesi), `staged_50` (kademeli birleştirme; yeni kol). Örnekler: 12
yeni 20.000 düğümlük örnek (6 düzgün, 6 kümeli; 240 s) ve 2 yeni 50.000 düğümlük örnek (600 s; betimsel).

Okumalar ve karar kuralı `experiments/PROTOCOL_v28.md` dosyasındadır; defter, protokol ve defterin kendisi GitHub'da
(main) kurulduğu biçimde değilse başlamaz. Önce depoyu commit ve push edin.

**Çalıştırma:** Çalışma zamanı türü **L4 GPU + Yüksek RAM** (12 vCPU, 53 GB) seçin ve bütün hücreleri sırayla
çalıştırın. GPU kullanılmaz; ortam, makine türü önceki koşularla aynı olsun diye seçilir. Tahmini süre: kurulum (LKH-3
ve GA-EAX derlemeleri) 5-7 dk; öz sınama 2-3 dk; 20.000 düğümlük 12 örnek, her biri yaklaşık 11,5 dk, 2,3 saat;
50.000 düğümlük 2 örnek, her biri yaklaşık 28 dk, 1 saat; çözümleme ve arşiv 1 dk; toplam yaklaşık 3,4 saat. Biten
örnekler Drive'a yazılır (`MyDrive/BetaTour/v28_staged14`) ve defter yeniden çalıştırılınca atlanır; makine değişirse
yarım kalan örnek baştan koşar. İlk hücredeki `MAX_NEW_INSTANCES` deneyi oturumlara böler. Sonunda indirilen
`v28_staged14_results.zip` dosyasını gönderin.
