# Run v30: kanıtlı optimuma ulaşma süresi

**Protokol:** `experiments/PROTOCOL_v30.md`. Koşudan önce yazıldı; defter, protokol ve defter GitHub'da değilse başlamaz.

**Soru.** Optimal kenarları daha az kaçıran listeler, LKH-3'ün kanıtlı optimuma daha çabuk ulaşmasını sağlar mı?

**Örnekler (37).** Pilotun 18 sentetik örneği (optimumları Concorde ile kanıtlandı) ve yayımlanmış optimumu olan 19 TSPLIB örneği (1.002–2.392 düğüm).

**Kollar (7):**

- Kesintisiz: `pop` (POPMUSIC-5) ve `alpha` (LKH-3'ün varsayılanı).
- İlk denemeden sonra yeniden başlatılan: `pop_r`, `beta`, `count`, `mix`, `union`.

Kol başına 6 tohum kullanılır. Üst sınır düğüm başına 0,05 s LKH-3 işlemci saatidir.

**Ölçüt.** PAR10: optimuma ulaşma süresi; sınırda ulaşamayan koşu 10 × sınır sayılır. İki kolun PAR10 farkı sınırın %1'ini aşmıyorsa eşit sayılır.

**Hipotezler** (kesin işaret testi, Holm):

- **H1:** `beta` < `pop`.
- **H2:** `beta` < `alpha`.

**Çalıştırma.**

1. Protokolü ve defteri commit edip GitHub'a push edin.
2. Colab'da L4 GPU + Yüksek RAM (12 vCPU) seçin; T4 + 8 vCPU da olur.
3. *Tümünü çalıştır*. İlk hücre Drive'ı bağlar. Sonuçlar `MyDrive/BetaTour/v30_tto37` klasörüne yazılır.
4. Oturum düşerse aynı makine türüyle yeniden çalıştırın; biten koşular atlanır.

**Tahmini süre:**

| Adım | 12 vCPU | 8 vCPU |
|---|---|---|
| Kurulum | 3–5 dk | 3–5 dk |
| Öz sınama | 1–2 dk | 1–2 dk |
| İlk denemeler ve listeler | 4–8 dk | 5–10 dk |
| Koşular (1.554) | 3–5 saat | 4–7 saat |
| Analiz ve arşiv | 1–2 dk | 1–2 dk |
