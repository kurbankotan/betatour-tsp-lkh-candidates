# BetaTour v10: çok kenarlı tur onarım pilotu

Bu çalışma **algoritma geliştirme pilotudur**. v8/v9'da incelenmiş sekiz EUC_2D örneğini kullanır; bağımsız doğrulama verisi değildir. Sonuçları olumlu olsa bile tek başına makalede üstünlük iddiası oluşturmaz.

## Soru ve yöntem

v9, ilk turun ardından tek seferde aday kenar ekledi; POPMUSIC-6 karşısında güvenilir bir kazanç göstermedi. v10, mevcut turu doğrudan değiştirebilen bir mekanizma sınar: üç sorunlu kesim çevresindeki tur pencerelerini serbest bırakır, kalan uzun parçaları zorunlu kenarlara sıkıştırır ve yaklaşık 3 × 256 serbest düğümlü küçük bir TSP'yi LKH ile çözer. Zorunlu kenarlar açılarak tam tur geri kurulur. Yalnızca geçerli ve kesin olarak daha kısa bir tur kabul edilir. Orijinal tur her zaman mümkündür. Bu, öğrenilmiş onarım seçicisine yatırım yapmadan önce çok kenarlı onarımın kısa Colab bütçesinde işe yarayıp yaramadığını ölçer.

Altı eşlenmiş POPMUSIC-6/LKH süreci bütçenin yaklaşık %35'inde, tamamlanan ilk uygun denemeden sonra v10 kancasında bekletilir. Böylece onarım, ilk tur yerine arama sırasında olgunlaşmış bir tura uygulanır. Onarımlar dış süreçte yapılır; aynı LKH süreçleri müdahalesiz devam eder. Sonuç, sürenin sonunda LKH'nin bulduğu tur ile onarılmış turun kısa olanıdır. Onarım ve model çıkarımı aynı 120/300 saniyelik duvar saati sınırına dahildir. Onarılmış tur LKH'nin iç durumuna enjekte edilmez; raporda yöntem açıkça **yan onarımlı hibrit** olarak adlandırılır.

## Karşılaştırmalar

| Kol | Rol |
| --- | --- |
| Kesintisiz POPMUSIC-6 | Pratik güçlü karşılaştırma. |
| Kancalı sham | Bekleme ve devam etkisini ölçer; onarım yapmaz. |
| Rastgele kesimler | Aynı onarım motoruyla seçici kontrolü. |
| Uzun/olağandışı kenar seçimi | Güçlü, modelsiz yerel sezgisel kontrol. |
| Altı turun uyuşmazlığı | Popülasyon bilgisini modelsiz kullanan kontrol. |
| BetaTour seçimi | Varsa mevcut TourFold puanıyla kesim seçimi; bu ağ henüz gerçek onarım faydasına göre eğitilmemiştir. |

Onarımlı kollar aynı kesim sayısı, deneme üst sınırı ve toplam süre payını kullanmalıdır. Model çıkarımı da BetaTour kolunun süre bütçesinden düşülür. POPMUSIC-6 gerçek kullanım karşılaştırmasıdır; yalnız başına yöntem etkisini izole etmez, çünkü yan onarım ek hesaplama kullanır. BetaTour'un seçici katkısı ancak **aynı yan onarım motorunu kullanan** kontrol kollarına karşı yorumlanır.

## Kayıt ve karar

Her tohum için bariyerdeki en iyi tur, LKH son turu, onarım son turu, seçilen kesimler, onarım denemeleri, maliyetler, zaman damgaları ve kaynak/binary/model hash'leri saklanır. Turların Hamilton geçerliliği ve TSPLIB EUC_2D tamsayı maliyeti koordinatlardan yeniden hesaplanır. İnceleme birimi sekiz **örnektir**, altı tohum bağımsız örnek sayılmaz. Eşit toplam süre içinde örnek başına en kısa geçerli tur, zaman–kalite eğrisi ve onarımın kendi kazancı raporlanır.

Pilot için karar sırası:

1. Onarım motoru geçerli, kesin iyileştiren turlar üretemiyorsa bu kısa bütçeli tasarımı durdur.
2. Modelsiz seçiciler fayda sağlıyor ama BetaTour sağlamıyorsa, mevcut kenar sınıflandırıcısını başarı diye sunma; gerçek onarım kazancı/saniye etiketleriyle durum koşullu bir seçici eğit.
3. BetaTour hem POPMUSIC-6'yı hem de güçlü modelsiz yan onarım kontrolünü tekrarlı biçimde geçerse kodu ve parametreleri dondurup yeni, bağımsız örneklerde daha geniş deney yap.

Makaleye yalnızca bağımsız deneyler olumlu sonuç verdikten sonra yeni performans iddiası eklenir. Olumsuz pilot, mevcut v9 makale taslağını değiştirmez.
