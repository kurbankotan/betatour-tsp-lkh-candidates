# BetaTour run v22: düzeltilmiş dosyalarla diğer boyutlar ve TSPLIB (genelleme koşusu)

**Neden bu koşu?** Run v21, 10.000 düğümlü 24 örnekte iki şey gösterdi: run v1–v8'in aday dosyaları yeniden başlatılan LKH-3'ü yavaşlatmıştı; LKH'ninkiyle eşdeğer dosyalarla BetaTour listeleri kesintisiz POPMUSIC'ten daha kısa tur buldu (24 örneğin 18'inde; tohum ortalamasında 22'sinde). Sayım listesi BetaTour kadar iyi çıktı. Protokol, bu sonucu makalede genel bir iddia olarak yazmadan önce diğer boyutlarda ve TSPLIB'de tekrarlamayı şart koşuyordu. Bu defter o tekrardır; kod ve kurallar run v21 ile aynıdır.

**Tasarım.** 43 örnek: 16 yeni 1.000 düğümlü (bütçe 5 s), 16 yeni 20.000 düğümlü (bütçe 300 s) ve önceki koşuların 11 TSPLIB örneği (pcb3038 … d18512; bütçe max(10, 0,02 n) s). Altı tohum (101–106), altı paralel LKH-3.0.14 süreci. Dört kol:

| Kol | Aday listesi | Dosya |
|---|---|---|
| `pop_full` | POPMUSIC-5, kesintisiz | yok |
| `pop_native` | aynı tohumun POPMUSIC-5 listesi, yeniden başlatma | LKH'nin kendi alfa ve baba değerleri |
| `beta_tree` | BetaTour karma listesi (model ilk 3 + POPMUSIC) | ağaç kodlaması, LKH'nin baba değerleri |
| `freq_tree` | sayım karma listesi | ağaç kodlaması, LKH'nin baba değerleri |

**Önceden belirlenen iki birincil karşılaştırma** (`experiments/PROTOCOL_v22.md`, koşudan önce yazıldı; bütçe sonunda, altı tohumun eşli ortalaması, 43 örnek üzerinde kesin iki yönlü işaret testi, ikisi birlikte Holm düzeltmeli):

- **P1:** `beta_tree` / `pop_full`. Düzeltilmiş dosyayla BetaTour, kesintisiz POPMUSIC'i bu boyutlarda da geçiyor mu?
- **P2:** `beta_tree` / `freq_tree`. Öğrenilmiş model sayımdan daha iyi mi?

Küçük örneklerde (1.000 düğüm, küçük TSPLIB) altı tohumun en iyisi büyük olasılıkla eşit çıkacak, çünkü LKH bunları bütçe içinde zaten çözüyor; orada fark, altı tohumdan kaçının en iyi tura ulaştığında görülür. Sonuç hangi yönde çıkarsa çıksın makaleye girer.

**Çalıştırma.** Colab'da **A100 veya L4 GPU** ve **Yüksek RAM** (12 vCPU) seçin, sonra *Çalışma zamanı → Tümünü çalıştır*. Drive izni istendiğinde onaylayın ve sekmeyi açık tutun. Bağlantı koparsa aynı ortam türüyle yeniden *Tümünü çalıştır* deyin: biten örnekler Drive'daki `MyDrive/BetaTour/v22_sizes43` klasöründen okunur ve atlanır; yarım kalan örnek baştan koşar.

**Tahmini süre.**

| Aşama | Süre |
|---|---|
| Kurulum, depo, LKH derlemesi, TSPLIB indirme | ≈ 4–6 dk |
| Isınma, öz sınama ve TSPLIB ön denetimi | ≈ 2 dk |
| 16 × 1.000 düğüm (örnek başına ≈ 25 s) | ≈ 7 dk |
| 16 × 20.000 düğüm (örnek başına ≈ 19 dk) | ≈ 5,1 saat |
| 11 TSPLIB örneği (örnek başına 4–24 dk) | ≈ 2,3 saat |
| Çözümleme ve ZIP | ≈ 1 dk |
| **Toplam** | **≈ 7,5 saat** |

Bölmek için ilk hücrede `MAX_NEW_INSTANCES = 26` yazın: ilk oturum 1.000 düğümlü örnekleri ve büyük örneklerin ilk bölümünü bitirir (≈ 2,5 saat), sonraki oturumlarda `MAX_NEW_INSTANCES = 9` ile devam edin (oturum başına ≈ 2,5 saat). Sonunda `v22_sizes43_results.zip` tarayıcıya iner (Drive'da da durur); bu dosyayı gönderin.
