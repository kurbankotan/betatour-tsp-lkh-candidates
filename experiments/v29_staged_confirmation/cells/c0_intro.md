# BetaTour run v29 (doğrulama koşusu): kademeli birleştirme

Run v28'de (pilot) altı LKH-3 sürecinin bütçenin yarısındaki 102 turu iki aşamada birleştirildi: önce altı grupta altı
GA-EAX süreci aynı anda, sonra grupların sonuç turları ve altı en iyi tur tek GA-EAX sürecinde. 20.000 düğümlü 12
örnekte bu kol, iki GA-EAX sürecinden (run v26'nın birincil kolu) 10 örnekte, LKH-3'ün kendi tur birleştirmesinden 12
örnekte daha kısa tur verdi. Bu koşu aynı ayarlarla, yeni örneklerde ve önceden yazılmış iki hipotezle doğrular:

- **H1:** kademeli birleştirme, LKH-3'ün kendi tur birleştirmesinden daha kısa tur verir.
- **H2:** kademeli birleştirme, iki GA-EAX sürecinden daha kısa tur verir.

Kollar (her örnekte tek bir kayıtlı koşudan; kesme anı bütçenin yarısı): `lkh6`, `trials_50`, `best6_50`, `dual_50`,
`lkhmerge_50`, `staged_50`. Örnekler: 18 yeni sentetik örnek (20.000 düğüm: 6 düzgün, 6 kümeli, 240 s; 50.000 düğüm:
3 düzgün, 3 kümeli, 600 s) hipotezler için ve optimumu bilinen 5 büyük TSPLIB örneği (betimsel).

Hipotezler ve karar kuralı `experiments/PROTOCOL_v29.md` dosyasındadır; defter, protokol ve defterin kendisi GitHub'da
(main) kurulduğu biçimde değilse başlamaz. Önce depoyu commit ve push edin.

**Çalıştırma:** Çalışma zamanı türü **L4 GPU + Yüksek RAM** (12 vCPU, 53 GB) seçin ve bütün hücreleri sırayla
çalıştırın. GPU kullanılmaz; ortam, makine türü önceki koşularla aynı olsun diye seçilir. Tahmini süre: kurulum (LKH-3
ve GA-EAX derlemeleri, TSPLIB dosyaları) 5-7 dk; öz sınama 2-3 dk; 20.000 düğümlük 12 örnek, her biri yaklaşık 11,5
dk, 2,3 saat; 50.000 düğümlük 6 örnek, her biri yaklaşık 28 dk, 2,8 saat; 5 TSPLIB örneği, her biri 11-17 dk, 1,1
saat; çözümleme ve arşiv 1 dk; toplam yaklaşık 6,3 saat. Biten örnekler Drive'a yazılır
(`MyDrive/BetaTour/v29_staged23`) ve defter yeniden çalıştırılınca atlanır; makine değişirse yarım kalan örnek baştan
koşar. Deneyi iki oturuma bölmek için ilk hücrede `MAX_NEW_INSTANCES = 12` yapın, ikinci oturumda yeniden çalıştırın.
Sonunda indirilen `v29_staged23_results.zip` dosyasını gönderin.
