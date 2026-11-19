# BetaTour run v21: aday dosyası kodlaması denetimi (düzeltilmiş ölçüm)

**Neden bu koşu?** Run v1–v8'de LKH-3 ilk denemeden sonra durduruluyor ve bir aday dosyasıyla yeniden başlatılıyordu. LKH-3.0.14'ün kaynak koduna göre bu dosyanın iki alanı aramayı etkiler ve o koşuların dosyaları ikisini de LKH'nin kendi dosyası gibi yazmamıştı:

1. **Alfa değerleri.** LKH-3 ilk denemeden sonraki her denemeye, en iyi turun kenarlarının bir bölümünü koruyarak başlar; bir kenar yalnızca aday listesinde **alfa değeri 0** ise korunur (`ChooseInitialTour.c`, C durumu). LKH kendi POPMUSIC listelerinde düğüm başına yaklaşık iki adaya alfa = 0 verir (en küçük 1-ağacın kenarları). Eski dosyalar alfa = 0, 100, 200, … ile yazılmıştı; yani düğüm başına yalnızca ilk adayın alfası 0'dı. Ön denemede eski dosyayla tur kenarlarının %44'ü, LKH'nin kendi değerleriyle %76'sı korunabiliyordu.
2. **Baba alanı (Dad).** Düğümün en küçük 1-ağaçtaki babası. İlk tur kaydedilene kadar LKH bu kenarları ilk adımda kırmaz (`LKH.h`, `Near`). Eski dosyalarda bu alan boştu.

Bu defter, bu farkın yeniden başlatılan kolları ne kadar etkilediğini ve LKH'ninkiyle eşdeğer dosyalarla BetaTour'un nerede durduğunu ölçer.

**Tasarım.** Daha önce hiçbir koşuda kullanılmamış 24 örnek (10.000 düğüm; 12 uniform, 12 kümeli), bütçe 120 s, altı tohum (101–106), altı paralel LKH-3.0.14 süreci. Altı kol:

| Kol | Aday listesi | Alfa değerleri | Baba alanı |
|---|---|---|---|
| `pop_full` | POPMUSIC-5, kesintisiz | LKH'nin kendisi (dosya yok) | LKH'nin kendisi |
| `pop_native` | aynı tohumun POPMUSIC-5 listesi, yeniden başlatma | LKH'nin kendi değerleri | LKH'nin kendi değerleri |
| `pop_rank` | aynı liste, yeniden başlatma | 0, 100, 200, … | boş (run v5'teki `restart_pop_raw`) |
| `beta_tree` | BetaTour karma listesi (model ilk 3 + POPMUSIC) | ağaç kodlaması | LKH'nin kendi değerleri |
| `beta_rank` | aynı BetaTour listesi | 0, 100, 200, … | boş (run v5'teki `betatour`) |
| `freq_tree` | sayım karma listesi | ağaç kodlaması | LKH'nin kendi değerleri |

*Ağaç kodlaması:* aynı tohumun kendi POPMUSIC listesinde alfası 0 olan kenarlar listede varsa alfa = 0, diğer girdiler liste sırasıyla 100, 200, … Zaman hesabı run v5 ile aynıdır: yeniden başlatılan kolların bütçesinden ortak ilk aşamanın ve listenin hazırlık süresi düşülür.

**Önceden belirlenen iki birincil karşılaştırma** (`experiments/PROTOCOL_v21.md`, koşudan önce yazıldı; bütçe sonunda, örnek düzeyinde kesin iki yönlü işaret testi, ikisi birlikte Holm düzeltmeli):

- **P1:** `pop_rank` / `pop_native` (altı tohumun eşli ortalaması). Eski dosya, yeniden başlatılan LKH'yi kötüleştirdi mi?
- **P2:** `beta_tree` / `pop_full` (altı tohumun en iyisi). Düzeltilmiş dosyayla BetaTour, kesintisiz POPMUSIC'i geçiyor mu?

Sonuç hangi yönde çıkarsa çıksın makaleye girer.

**Çalıştırma.** Colab'da **A100 veya L4 GPU** ve **Yüksek RAM** (12 vCPU) seçin, sonra *Çalışma zamanı → Tümünü çalıştır*. Drive izni istendiğinde onaylayın ve sekmeyi açık tutun. Bağlantı koparsa aynı ortam türüyle yeniden *Tümünü çalıştır* deyin: biten örnekler Drive'daki `MyDrive/BetaTour/v21_alphaenc24` klasöründen okunur ve atlanır; yarım kalan örnek baştan koşar.

**Tahmini süre.**

| Aşama | Süre |
|---|---|
| Kurulum, depo, LKH derlemesi | ≈ 3–5 dk |
| Isınma ve kodlama öz sınaması | ≈ 1 dk |
| Örnek başına: ortak ilk aşama ≈ 15 s, kesintisiz kol ≈ 2 dk, beş yeniden başlatılan kol ≈ 9 dk | ≈ 11,5 dk |
| 24 örnek | **≈ 4,6 saat** |
| Çözümleme ve ZIP | ≈ 1 dk |

Daha kısa oturumlar için ilk hücrede `MAX_NEW_INSTANCES = 8` yazın (≈ 1,5 saat); defteri üç kez çalıştırmak koşuyu tamamlar. Sonunda `v21_alphaenc24_results.zip` tarayıcıya iner (Drive'da da durur); bu dosyayı gönderin.
