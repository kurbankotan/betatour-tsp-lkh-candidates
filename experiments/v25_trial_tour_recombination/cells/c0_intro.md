# BetaTour run v25 (pilot): LKH'nin attığı deneme turlarını birleştirmek

**Fikir.** LKH-3 her denemede bir yerel en iyi tur üretir, bunu en iyi turuyla birleştirir ve yalnızca en iyi turu tutar; deneme turunun kendisi atılır. 20 Kasım 2026 ön denemesinde (tek örnek, 10.000 düğüm, geliştirme ortamı) altı sürecin son 17'şer deneme turunu GA-EAX ile birleştirmek, altı sürecin en iyisinden %0,023 daha kısa tur verdi; yalnızca altı en iyi turu birleştirmek %0,013 verdi. 50.000 düğümlü tek örnekte fark %0,06 dolayındaydı, ama o karşılaştırma eşit sürede yapılmamıştı. Bu pilot aynı şeyi yeni örneklerde ve **eşit duvar saati süresinde** ölçer.

**Bu bir pilottur (keşif amaçlı).** Sonuç iyi de çıksa makaleye girecek bir iddia için, kuralı önceden yazılmış ve çalıştırılmadan önce depoya gönderilmiş ayrı bir doğrulama koşusu gerekir.

**Kollar.** Her kolun toplam süresi örneğin bütçesidir ve hepsi aynı makinede koşar. Bütün zamanlar duvar saatidir.

| Kol | Ne yapar |
|---|---|
| `lkh6` | Altı LKH-3.0.14 süreci (POPMUSIC-5, tohum 101–106) bütçenin sonuna kadar sürer; altısının en iyisi. Önceki koşulardaki `pop_full` ayarları. |
| `trials_50` (birincil) | Aynı altı süreç bütçenin yarısında durur. Her sürecin en iyi turu ve son 16 atılan deneme turu (102 tur) GA-EAX'in popülasyonu olur; GA-EAX kalan sürede tek çekirdekte koşar. |
| `best6_50` (kontrol) | Aynısı, ama popülasyon yalnızca altı en iyi turdur. Atılan turların katkısını ayırır. |
| `trials_33`, `trials_67`, `best6_33`, `best6_67` | Aynı iki kol, LKH bütçenin üçte birinde ya da üçte ikisinde durdurulursa. |
| `cascade` | Bilinen melez: altı işçi LKH'nin tek denemelik 100 turunu üretir, GA-EAX bu turlardan başlar ve bütçe sonunda durur. |
| `eax_scratch` (referans) | Sıfırdan GA-EAX, yazarın varsayılan ayarları (popülasyon 300), tek süreç, 6 × bütçe saniye (altı LKH sürecinin toplam CPU süresi). Yalnızca 20.000 düğüme kadar. |

`lkh6` ile birleştirme kolları aynı kayıtlı koşudan çıkar: LKH'nin kayıt eden kopyası her denemenin turunu ve her yeni en iyi turu duvar saati damgasıyla dosyaya yazar; arama değişmez (öz sınama bunu bu makinede denetler). Birleştirme kollarında GA-EAX'e verilen süre = bütçe − LKH'nin durdurulduğu an − popülasyonu hazırlama süresi. GA-EAX tek çekirdekte koşar; LKH durduktan sonra diğer beş çekirdek boş kalır.

**Örnekler (21).** Daha önce kullanılmamış 12 örnek 10.000 düğüm (6 uniform, 6 kümeli; bütçe 120 s), optimumu bilinen 5 büyük TSPLIB örneği (rl11849, usa13509, brd14051, d15112, d18512; bütçe 0,02 × düğüm sayısı), 4 örnek 50.000 düğüm (2 uniform, 2 kümeli; bütçe 600 s).

**Önceden yazılan karar kuralı** (`experiments/PROTOCOL_v25.md`):

- **D1 (tekrar):** 10.000 düğümlü 12 örnekte `trials_50`, `lkh6`'dan daha çok örnekte kısa ve kesin iki yönlü işaret testi p < 0,05 (eşitlik yoksa en az 10 örnek).
- **D2 (boyutla büyüme):** 50.000 düğümlü dört örneğin hepsinde `trials_50` daha kısa ve ortalama fark 10.000 düğümdekinden büyük.
- D1 ve D2 birlikte sağlanırsa doğrulama koşusu hazırlanır. Sağlanmazsa sonuç depoya kısa bir kayıt olarak yazılır.
- D3 (bilgi amaçlı): 10.000 düğümde `trials_50`, `best6_50`'den daha çok örnekte kısa ve p < 0,05.

**Çalıştırma.** Colab'da **L4 GPU** ve **Yüksek RAM** seçin (12 vCPU, 53 GB; GPU kullanılmaz, bu ortam yalnızca çekirdek sayısı ve bellek için gerekir; A100 de olur ama gereksiz yere pahalıdır). Sonra *Çalışma zamanı → Tümünü çalıştır*. Drive izni istendiğinde onaylayın ve sekmeyi açık tutun. Bağlantı koparsa aynı ortam türüyle yeniden *Tümünü çalıştır* deyin: biten örnekler Drive'daki `MyDrive/BetaTour/v25_trialtours21` klasöründen okunur ve atlanır; yarım kalan örnek baştan koşar.

**Tahmini süre.**

| Aşama | Süre |
|---|---|
| Kurulum: depo, LKH'nin iki derlemesi, GA-EAX'in iki derlemesi, TSPLIB dosyaları | ≈ 4–6 dk |
| Öz sınama | ≈ 1 dk |
| 10.000 düğüm, örnek başına ≈ 6 dk (kayıtlı LKH 2 dk, birleştirmeler 1,3 dk, bilinen melez 2 dk) × 12 | ≈ 1,2 saat |
| TSPLIB, örnek başına ≈ 11–17 dk × 5 | ≈ 1,1 saat |
| 50.000 düğüm, örnek başına ≈ 28 dk (kayıtlı LKH 10 dk, birleştirmeler 6,7 dk, bilinen melez 10 dk) × 4 | ≈ 1,9 saat |
| Sıfırdan GA-EAX referansı (altışar örnek paralel; 12 + 12 + 37 dk) | ≈ 1,0 saat |
| Çözümleme ve ZIP | ≈ 1 dk |
| **Toplam** | **≈ 5,3 saat** |

İlk karar (D1) yaklaşık 1,3 saat sonra, ilk 12 örnek bitince ara raporda görülür. Daha kısa oturumlar için ilk hücrede `MAX_NEW_INSTANCES = 12` yazın (≈ 1,3 saat); defteri yeniden çalıştırmak kalan örneklerle sürer. Referans koşuları ana örneklerin hepsi bitince başlar; istemezseniz `RUN_REFERENCE = False` yazın (≈ 1 saat kısalır). Sonunda `v25_trialtours21_results.zip` tarayıcıya iner (Drive'da da durur); bu dosyayı gönderin.
