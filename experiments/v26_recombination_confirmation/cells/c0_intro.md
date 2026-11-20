# BetaTour run v26: doğrulama koşusu (LKH-3 turlarının GA-EAX ile birleştirilmesi)

**Ne ölçülüyor.** Pilot (run v25) önceden yazılan kuralını sağladı: altı LKH-3 süreci bütçenin yarısında durdurulup turları GA-EAX ile birleştirilince, 21 örneğin 21'inde süreçleri sonuna kadar koşturmaktan daha kısa tur çıktı. Bu koşu aynı fikri **yeni örneklerde**, **koşudan önce GitHub'a gönderilmiş bir protokolle** (`experiments/PROTOCOL_v26.md`) ve iki yeni kontrolle sınar. Pilotta sonuçlar görüldükten sonra seçilen birleşim (iki popülasyonun kısa olanı) burada önceden sabitlenmiş birincil koldur.

**Önce GitHub.** Bu defter, protokol ve defterin kendisi GitHub'daki `main` dalında yoksa ya da farklıysa **çalışmaya başlamaz**. Önce depodaki değişiklikleri commit ve push edin, sonra bu defteri çalıştırın. Defter, protokolün commit'ini ve zamanını ilk örnekten önce manifeste yazar.

**Kollar.** Her kolun toplam süresi örneğin bütçesidir; bir örneğin bütün kolları aynı makinede, art arda koşar. Bütün zamanlar duvar saatidir.

| Kol | Ne yapar |
|---|---|
| `lkh6` | Altı LKH-3.0.14 süreci (POPMUSIC-5, tohum 201–206) bütçenin sonuna kadar sürer; altısının en iyisi. Süreçler her denemenin turunu kaydeder. |
| `dual_50` (birincil) | Aynı altı süreç bütçenin yarısında durur. İki GA-EAX süreci aynı anda kalan sürede koşar: biri 102 turla (her sürecin en iyi turu + son 16 atılan deneme turu), öteki altı en iyi turla. Sonuç ikisinden kısa olanıdır. |
| `trials_50`, `best6_50` | Birincil kolun iki parçası ayrı ayrı. |
| `lkhmerge_50` (yeni kontrol) | Altı en iyi tur, bütçenin yarısında LKH-3'ün kendi tur birleştirmesine (`MERGE_TOUR_FILE`) verilir; altı yeni LKH süreci kalan sürede arar. GA-EAX gerekli mi sorusunu yanıtlar. |
| `cascade` | Bilinen melez: LKH'nin tek denemelik 100 turu, sonra GA-EAX. |
| `eax_scratch` (referans) | Sıfırdan GA-EAX, tek süreç, 6 × bütçe saniye. Yalnızca 10.000 düğüm ve küçük TSPLIB örnekleri. |

**Örnekler (38).** Daha önce kullanılmamış 28 sentetik örnek: 16 × 10.000 düğüm (bütçe 120 s), 6 × 20.000 (240 s), 6 × 50.000 (600 s); yarısı uniform, yarısı kümeli. Optimumu bilinen 10 TSPLIB örneği (bütçe 0,02 × düğüm sayısı): pcb3038, fl3795, fnl4461, rl5915, rl5934 ve pilotun beş örneği (rl11849, usa13509, brd14051, d15112, d18512; yeni tohumlarla). TSPLIB örnekleri testlere girmez.

**Önceden yazılan testler** (28 sentetik örnek; kesin iki yönlü işaret testi, dört test üzerinden Holm düzeltmesi, alfa 0,05):

- **H1:** `dual_50`, `lkh6`'dan daha kısa.
- **H2:** `dual_50`, `lkhmerge_50`'den daha kısa (LKH'nin kendi birleştirmesi aynı işi görmüyor).
- **H3:** `dual_50`, `cascade`'den daha kısa.
- **H4:** 10.000 düğümlü 16 örnekte `trials_50`, `best6_50`'den daha kısa (atılan deneme turlarının katkısı).

**Çalıştırma.** Colab'da **L4 GPU** ve **Yüksek RAM** seçin (12 vCPU, 53 GB; run v25 ile aynı ortam; GPU kullanılmaz). Sonra *Çalışma zamanı → Tümünü çalıştır*. Drive izni istendiğinde onaylayın ve sekmeyi açık tutun. Bağlantı koparsa aynı ortam türüyle yeniden *Tümünü çalıştır* deyin: biten örnekler Drive'daki `MyDrive/BetaTour/v26_recomb38` klasöründen okunur ve atlanır; yarım kalan örnek baştan koşar.

**Tahmini süre.**

| Aşama | Süre |
|---|---|
| Kurulum: depo ve protokol denetimi, LKH'nin iki derlemesi, GA-EAX'in dört derlemesi, 10 TSPLIB dosyası | ≈ 5–7 dk |
| Öz sınama | ≈ 2 dk |
| 10.000 düğüm, örnek başına ≈ 6 dk (kayıtlı LKH 2 dk, iki GA-EAX ≤ 1 dk, LKH birleştirmesi 1 dk, bilinen melez 2 dk) × 16 | ≈ 1,6 saat |
| 20.000 düğüm, örnek başına ≈ 12 dk × 6 | ≈ 1,2 saat |
| 50.000 düğüm, örnek başına ≈ 31 dk (kayıtlı LKH 10 dk, iki GA-EAX 5 dk, LKH birleştirmesi 5 dk, bilinen melez 10 dk) × 6 | ≈ 3,1 saat |
| TSPLIB, küçük beş örnek (3.000–6.000 düğüm) | ≈ 0,4 saat |
| TSPLIB, pilotun beş örneği (12.000–18.500 düğüm), örnek başına 12–19 dk | ≈ 1,2 saat |
| Sıfırdan GA-EAX referansı (21 örnek, altışar paralel) | ≈ 0,8 saat |
| Çözümleme ve ZIP | ≈ 1 dk |
| **Toplam** | **≈ 8,5 saat** (en çok ≈ 9 saat) |

Tek oturum uzun gelirse ilk hücrede `MAX_NEW_INSTANCES` ile bölün: `22` yazarsanız ilk oturum 10.000 ve 20.000 düğümlü örneklerle biter (≈ 3 saat); ikinci oturumda `6` (50.000 düğüm, ≈ 3,1 saat); üçüncüsünde `None` (TSPLIB ve referans, ≈ 2,5 saat). Her oturumda aynı ortam türünü seçin. Sonunda `v26_recomb38_results.zip` tarayıcıya iner (Drive'da da durur); bu dosyayı gönderin.
