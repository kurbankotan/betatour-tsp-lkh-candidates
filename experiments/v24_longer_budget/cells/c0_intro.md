# BetaTour run v24: 20.000 düğümde iki katı bütçe (doğrulayıcı koşu)

**Neden bu koşu?** Düzeltilmiş aday dosyalarıyla BetaTour, 10.000 düğümde kesintisiz POPMUSIC'ten daha kısa tur buldu (run v21: 24 örneğin 18'i; run v23: 12 yeni örneğin 10'u). Run v22'de aynı karşılaştırma 20.000 düğümde fark göstermedi (16 örnekte 8'e 8; bütçe 300 s). Run v22'nin zaman seyri ise bir eğilim gösteriyordu: uniform örneklerde BetaTour bütçenin çeyreğinde %0,016 gerideydi, bütçe sonunda %0,001 öndeydi. Yeniden başlatmanın kendi maliyeti de (eşdeğer dosya / kesintisiz, altı tohum ortalaması) 20.000 düğümde bütçe sonunda hâlâ +%0,004 idi; 10.000 düğümlü uniform örneklerde bütçenin yarısında +%0,001'e inmişti. Bu koşu şunu sorar: 20.000 düğümde kazanç yok mu, yoksa yalnızca daha geç mi geliyor?

**Tasarım.** Daha önce hiçbir koşuda kullanılmamış 20 örnek (20.000 düğüm; 10 uniform, 10 kümeli), bütçe 600 s (run v22'deki bütçenin iki katı), altı tohum (101–106), altı paralel LKH-3.0.14 süreci. Üç kol:

| Kol | Aday listesi | Alfa değerleri ve baba alanı |
|---|---|---|
| `pop_full` | POPMUSIC-5, kesintisiz | LKH'nin kendisi (dosya yok) |
| `pop_native` | aynı tohumun POPMUSIC-5 listesi, yeniden başlatma | LKH'nin kendi değerleri |
| `beta_tree` | BetaTour karma listesi (model ilk 3 + POPMUSIC), yeniden başlatma | ağaç kodlaması, LKH'nin baba alanı |

Kollar, dosyalar ve zaman hesabı run v21 ve v22 ile aynıdır (aynı modül). Tek fark dış süre sınırıdır: run v22'de süreçler bütçeden 3 s sonra durduruluyordu ve 20.000 düğümde LKH'nin saati (sürecin CPU süresi) duvar saatinden yaklaşık %1,5 geri kaldığı için bütçenin son saniyeleri gözlenmemişti. Burada sınır bütçeden 60 s sonradır (bütçenin %10'u) ve yalnızca güvenlik içindir: süreçler LKH'nin kendi süre sınırında biter. Rapor, bütçenin sonu gözlenmeyen süreç olup olmadığını ayrıca yazar.

**Önceden belirlenen tek birincil karşılaştırma** (`experiments/PROTOCOL_v24.md`, koşudan önce yazıldı):

- **P1:** `beta_tree` / `pop_full`, bütçe sonunda (600 s), altı tohumun eşli ortalaması; 20 örnek üzerinde kesin iki yönlü işaret testi (p < 0,05; eşitlik yoksa en az 15 örnek).

İkincil (keşif amaçlı): aynı karşılaştırma 300 s'de (run v22'nin bütçesi; yeni örneklerde tekrar), yeniden başlatmanın maliyeti (`pop_native` / `pop_full`), listenin etkisi (`beta_tree` / `pop_native`), zaman seyri ve aileler. Sonuç hangi yönde çıkarsa çıksın makaleye girer.

**Çalıştırmadan önce** `experiments/PROTOCOL_v24.md` dosyasını depoya commit edip gönderin (push). Run v22 ve v23'te bu adım atlanmıştı ve protokollerin kamuya açık bir zaman damgası olmadı.

**Çalıştırma.** Colab'da **L4 (ya da A100) GPU** ve **Yüksek RAM** (12 vCPU) seçin, sonra *Çalışma zamanı → Tümünü çalıştır*. Drive izni istendiğinde onaylayın ve sekmeyi açık tutun. Bağlantı koparsa aynı ortam türüyle yeniden *Tümünü çalıştır* deyin: biten örnekler Drive'daki `MyDrive/BetaTour/v24_long20k20` klasöründen okunur ve atlanır; yarım kalan örnek baştan koşar.

**Tahmini süre.**

| Aşama | Süre |
|---|---|
| Kurulum, depo, LKH derlemesi | ≈ 3–5 dk |
| Isınma ve kodlama öz sınaması | ≈ 1–2 dk |
| Örnek başına: ortak ilk aşama ≈ 33 s, kesintisiz kol ≈ 10,4 dk, iki yeniden başlatılan kol ≈ 19,3 dk | ≈ 30 dk |
| 20 örnek | **≈ 10 saat** |
| Çözümleme ve ZIP | ≈ 1 dk |

Daha kısa oturumlar için ilk hücrede `MAX_NEW_INSTANCES = 7` yazın (≈ 3,5 saat); defteri üç kez çalıştırmak koşuyu tamamlar (`10` yazarsanız iki oturum, her biri ≈ 5 saat). Sonunda `v24_long20k20_results.zip` tarayıcıya iner (Drive'da da durur); bu dosyayı gönderin.
