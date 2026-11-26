# Kanıtlı aday grafı: karar pilotu

Bu defter BetaTour çalışmasının bir koşusu değildir; yeni projenin ilk denemesidir.

**Soru.** Bir model seyrek aday listeleri önerir. Listelerin tamlığını bir alt sınır sertifikası kanıtlayabilir mi? Bir kenar ancak onu kullanan her tur bilinen bir turdan kanıtla uzunsa elenir. Kalan kenarlar her optimal turu içeren bir graf oluşturur. Bir düğümün listesi, düğümün kalan bütün kenarlarını içeriyorsa *kanıtla tamdır*.

**Sertifikalar.**

- **C1:** LKH-3'ün düğüm cezalarıyla 1-ağaç sınırı ve bütün kenarların alfa değerleri. Kendi kodumuz, tam sayı aritmetiği.
- **C2:** Concorde'un kesme düzlemli kök LP'si: alt sınır ve `concorde -z` ile yazılan indirgenmiş maliyetler. Yalnızca kanıtlı optimuma karşı denetimleri geçerse kullanılır.

**Örnekler.** Çalışmanın üreticisiyle 20 yeni örnek: 1.000 ve 2.000 düğüm, tekdüze ve kümeli, beşer tane. Ayrıca optimumu bilinen 4 TSPLIB örneği. Concorde her örneği kanıtlı optimuma kadar çözer.

**Karar kuralı (pilottan önce yazıldı).**

- **G1:** C2, kanıtlı optimumu olan her örnekte denetimleri geçer.
- **G2:** C2 ve ÜS = LKH turu ile kalan grafın ortalama derecesi her sentetik grupta en çok 8'dir. Kısmi dökümde karar verilemeyen düğümlerin payı en çok %1 olmalıdır.
- **G3:** BetaTour'un 5'li listesi kanıtla tam olan düğümlerin payında hem LKH'nin alfa-5 listesini hem sayma listesini, her iki boyutta en az 5 puan aşar.

**Çalıştırma.** Ortam olarak L4 GPU + Yüksek RAM (12 vCPU) seçin. Sonra *Çalışma zamanı → Tümünü çalıştır*. İlk hücre Drive'ı bağlar. Sonuçlar `MyDrive/BetaTour/certpilot1` klasörüne yazılır. Oturum düşerse defteri yeniden çalıştırın; biten adımlar atlanır.

**Tahmini süre (12 vCPU).**

| Adım | Süre |
|---|---|
| Kurulum (LKH, Concorde, QSopt, depo, model) | 5-8 dk |
| Öz sınama | 1-2 dk |
| LKH | 2-4 dk |
| BetaTour | ~1 dk |
| Concorde (kök LP ve kanıtlı optimum) | 30-60 dk |
| Analiz ve özet | 2-4 dk |
| **Toplam** | **yaklaşık 45-80 dk** |
