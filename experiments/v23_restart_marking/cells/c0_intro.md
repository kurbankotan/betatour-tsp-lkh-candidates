# BetaTour run v23: yeniden başlatma işaretlemesi (keşif koşusu)

**Fikir.** LKH-3 ilk denemeden sonraki her denemeye en iyi turun büyük bölümünü koruyarak başlar. Bir tur kenarı yalnızca aday dosyasında **alfa değeri 0** ise korunur (`ChooseInitialTour.c`, C durumu); korunmayan yerlerde tur rastgele bozulur ve o denemenin araması yalnızca bu yerlerden başlayabilir (`LinKernighan.c`, kısıtlı arama). Alfa = 0 işaretleri bu yüzden her denemede turun neresinin açılacağını belirler. LKH bu işaretleri en küçük 1-ağaca göre koyar; run v21'in düzeltilmiş dosyaları da aynı kuralı kullandı.

Bu koşu, korunacak kenarları tur topluluğunun seçmesini dener:

- **Sayım işaretlemesi:** altı ilk deneme turunun en az beşinde bulunan kenarlar korunur.
- **Model işaretlemesi:** modelin tahmin olasılığı bir eşiğin üstünde olan kenarlar korunur. Eşik tohum başına seçilir: tohumun ilk turunda sayım işaretlemesiyle aynı sayıda kenar korunur. İki işaretleme böylece yalnızca *hangi* kenarları seçtiğinde ayrılır.

**Elimizdeki kanıt zayıftır.** Geliştirme ortamında 10.000 düğümlü üç örnekte LKH'nin kuralı ilk turun kenarlarının %24–25'ini açıkta bıraktı; turun, 150 saniyelik ayrı bir koşunun bulduğu daha iyi turda bulunmayan kenarlarının %59–62'si bu açıkta kalanlar arasındaydı. Sayım işaretlemesi kenarların %16–19'unu açıkta bırakıp bu kenarların %78–83'ünü, model işaretlemesi aynı payda %72–81'ini yakaladı. Aynı ortamda LKH ile yapılan küçük ön deneme (8 tek tohumlu koşu, 60 s) ise belirgin bir kazanç göstermedi (model eşikli işaretleme −%0,009; 5 koşuda kısa, 3 koşuda uzun; gürültü düzeyinde). Bu nedenle koşu **keşif amaçlıdır**: bir işaretleme daha iyi görünürse, makalede yer almadan önce ayrı bir doğrulama koşusunda sınanır. Sonuç hangi yönde çıkarsa çıksın kaydedilir.

**Tasarım.** Daha önce hiçbir koşuda kullanılmamış 12 örnek (10.000 düğüm; 6 uniform, 6 kümeli), bütçe 120 s, altı tohum (101–106), altı paralel LKH-3.0.14 süreci. Altı kol (iki liste × iki işaretleme ve BetaTour listesinde sayım):

| Kol | Aday listesi | Alfa = 0 işaretleri |
|---|---|---|
| `pop_full` | POPMUSIC-5, kesintisiz | LKH'nin kendisi (dosya yok) |
| `pop_native` | aynı tohumun POPMUSIC-5 listesi, yeniden başlatma | LKH'nin 1-ağacı |
| `pop_model` | aynı liste, yeniden başlatma | model işaretlemesi |
| `beta_tree` | BetaTour karma listesi (model ilk 3 + POPMUSIC) | LKH'nin 1-ağacı (run v21'deki kol) |
| `beta_model` | aynı BetaTour listesi | model işaretlemesi |
| `beta_count` | aynı BetaTour listesi | sayım işaretlemesi |

Bütün dosyalarda baba (Dad) alanı aynı tohumun kendi dosyasından gelir. Zaman hesabı run v21 ile aynıdır: yeniden başlatılan kolların bütçesinden ortak ilk aşamanın ve dosyanın hazırlık süresi (model, liste, işaretleme) düşülür.

**Önceden yazılan üç ana okuma** (`experiments/PROTOCOL_v23.md`, koşudan önce yazıldı; bütçe sonunda, altı tohumun eşli ortalaması):

- **M1:** `beta_model` / `beta_tree`. Model işaretlemesi BetaTour'u iyileştiriyor mu?
- **M2:** `beta_count` / `beta_tree`. Sayım işaretlemesi iyileştiriyor mu?
- **M3:** `pop_model` / `pop_native`. İşaretleme LKH'nin kendi listelerinde de işe yarıyor mu?

Karar kuralı: bir işaretleme, örnek düzeyinde kesin iki yönlü işaret testinde p < 0,05 ile daha kısa çıkarsa (eşitlik yoksa 12 örneğin en az 10'unda) onun için doğrulama koşusu hazırlanır; değilse fikir bu boyutta desteklenmemiş sayılır.

**Çalıştırma.** Colab'da **A100 veya L4 GPU** ve **Yüksek RAM** (12 vCPU) seçin, sonra *Çalışma zamanı → Tümünü çalıştır*. Drive izni istendiğinde onaylayın ve sekmeyi açık tutun. Bağlantı koparsa aynı ortam türüyle yeniden *Tümünü çalıştır* deyin: biten örnekler Drive'daki `MyDrive/BetaTour/v23_marking12` klasöründen okunur ve atlanır; yarım kalan örnek baştan koşar.

**Tahmini süre.**

| Aşama | Süre |
|---|---|
| Kurulum, depo, LKH derlemesi | ≈ 3–5 dk |
| Isınma ve kodlama öz sınaması | ≈ 1 dk |
| Örnek başına: ortak ilk aşama ≈ 15 s, kesintisiz kol ≈ 2 dk, beş yeniden başlatılan kol ≈ 9 dk | ≈ 11,5 dk |
| 12 örnek | **≈ 2,3 saat** |
| Çözümleme ve ZIP | ≈ 1 dk |

Daha kısa oturumlar için ilk hücrede `MAX_NEW_INSTANCES = 6` yazın (≈ 1,2 saat); defteri iki kez çalıştırmak koşuyu tamamlar. Sonunda `v23_marking12_results.zip` tarayıcıya iner (Drive'da da durur); bu dosyayı gönderin.
