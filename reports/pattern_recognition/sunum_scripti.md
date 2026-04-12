# Sunum Scripti - 15 Dakika

> Her slayt icin soylenmesi gerekenler, zamanlama ve demo plani.
> Toplam: ~15 dakika (10 dk slaytlar + 5 dk canli demo)

---

## SLAYT 1 - Baslik (0:00 - 0:30) [30 sn]

> "Merhaba hocam, ben Oruc Cakir, 211101023. Projemin adi: Coklu Ajan Destekli Aciklanabilir Yapay Zeka Metin Siniflandirma. Kisaca soyle ozetleyebilirim: 8 farkli makine ogrenmesi modelini 4 farkli veri setinde kiyaslayan, 3 ajanli bir pipeline uzerinde calisan ve LIME, SHAP gibi XAI yontemleriyle tahminleri ACIKLAYAN bir sistem gelistirdim. Hem Turkce hem Ingilizce destekliyor."

---

## SLAYT 2 - Problem (0:30 - 1:30) [1 dk]

> "Motivasyondan baslayalim. Gunumuzde metin siniflandirma modelleri yuksek dogruluk sagliyor, ama kara kutu gibi calisiyorlar - neden o karari verdiklerini aciklayamiyorlar. Bu projede uc temel soruya cevap ariyorum:"

> "Birincisi, farkli diller ve gorevler icin hangi model en iyisi? Turkce ve Ingilizce, sentiment ve haber siniflandirma gibi farkli senaryolari test ettim."

> "Ikincisi, klasik ML modelleri transformer'lara karsi nasil performans gosteriyor? SVM mi yoksa BERT mi daha iyi ve ne kadar farkla?"

> "Ucuncusu, model kararlarini LIME ve SHAP ile aciklayabilir miyiz? Kullanici modelin neden pozitif dediklerini gormeli."

> "Arastirma sorumuz burada: 4 veri seti, 8 model, 3 XAI yontemi ile kapsamli bir calisma."

---

## SLAYT 3 - 3 Ajanli Mimari (1:30 - 3:00) [1.5 dk]

> "Sistemin kalbi 3 ajanli bir pipeline. Bu mimariyi neden sectim? Cunku her ajan tek bir isten sorumlu ve bagimsiz calisiyor, modular bir yapi."

> "Ajan 1: Niyet Siniflandirici. Kullanici bir metin giriyor, bu ajan once dilin Turkce mi Ingilizce mi oldugunu tespit ediyor, sonra domain'i belirliyor - yani sentiment mi haber mi. Bunu Google Gemini LLM ile yapiyor, API yoksa Turkce karakter ve keyword tabanli fallback var."

> "Ajan 2: Siniflandirma Ajani. Ilk ajandan gelen bilgiyle dogru veri setini ve modeli seciyor. TF-IDF ile oznitelik cikartiyor, secilen modelle tahmin yapiyor, guven skoru donduruyor."

> "Ajan 3: XAI Ajani. Bu en kritik ajan. Tahmin sonucunu alip 3 farkli yontemle acikliyor: TF-IDF word importance, LIME ile lokal aciklama, SHAP ile Shapley degerleri. Sonra Gemini LLM ile bunlari dogal dile ceviriyor."

> "Kodda gordugunuz gibi pipeline.py'de bu 3 ajan sirayla calistirilir. process() metodu zincirleme cagirir."

---

## SLAYT 4 - Veri Setleri (3:00 - 3:45) [45 sn]

> "4 benchmark veri seti kullandim. Ikisi Ingilizce: IMDB film yorumlari, binary sentiment; AG News, 4 sinifli haber siniflandirma."

> "Ikisi Turkce: Turkish Sentiment, 440 bin kisa metin, 3 sinif. Bu en zorlayici veri seti cunku ciddi sinif dengesizligi var - negatif sinif sadece yuzde 11.6. Dorduncusu TTC4900, Turkce haber siniflandirma, 7 sinif."

> "Ana odak Turkish Sentiment cunku hem dengesiz hem Turkce morfoloji zorlugu var."

---

## SLAYT 5 - On Isleme ve TF-IDF (3:45 - 4:45) [1 dk]

> "On isleme pipeline'i 5 adimdan olusuyor: URL temizleme, kucuk harf donusumu, noktalama kaldir, stopword filtrele, minimum kelime uzunlugu kontrolu."

> "Turkce icin ozel bir sey: Turkce karakterleri - c cedilla, g breve gibi - koruyarak noktalama temizliyoruz."

> "Oznitelik cikariminda TF-IDF kullandim. 30 bin boyutlu sparse vektor, unigram + bigram, sublinear TF ve L2 normalizasyon. Bigram onemli cunku Turkce agglutinative bir dil - yani eklemeli. 'gel', 'geldi', 'gelmedi' gibi kelimeler var, bigram bu morfolojik varyasyonlari kismen yakaliyor."

> "Sagdaki t-SNE gorseli 2000 test ornegini gosteriyor. Gordugunuz gibi siniflar net olarak ayrilmiyor - bu bize supervised ogrenmein neden gerekli oldugunu gosteriyor."

---

## SLAYT 6 - Korelasyon ve Kumeleme (4:45 - 5:15) [30 sn]

> "Oznitelik analizinde Pearson korelasyon matrisi ve K-Means kumelemeyi test ettim. Korelasyon matrisinde top-15 SVM feature'i arasinda neredeyse hic korelasyon yok. Bu mantikli cunku TF-IDF sparse vektor uretiyor."

> "K-Means'in ARI skoru eksi 0.017 - yani rastgeleden bile kotu. Bu bize soyluyor ki: TF-IDF uzayinda unsupervised kumeleme ise yaramiyor, supervised siniflandiricilar sart."

---

## SLAYT 7 - Kovaryans Matrisi (5:15 - 5:45) [30 sn]

> "Kovaryans matrisine baktigimizda da ayni resmi goruyoruz: feature ciftleri arasinda kovaryans yaklasik sifir. TF-IDF sparse yapisi ve L2 normalizasyon varyansi sikistiriyor. Bu SVM icin avantaj, cunku SVM bolen duzlemi bagimsiz boyutlarda daha kolay buluyor."

---

## SLAYT 8 - Sinif Dagilimi ve Kelimeler (5:45 - 6:15) [30 sn]

> "Sinif dagilimina bakarsak: pozitif yuzde 51.5, notr yuzde 36.9, negatif sadece yuzde 11.6. Bu 1'e 3.2'ye 4.4 oraninda bir dengesizlik. Bu yuzden accuracy yaniltici, Macro-F1 asil metrigimiz."

> "SVM'in ogrendigi ayirt edici kelimelere bakarsak: pozitif icin 'harika', 'mukemmel', 'guzel'; negatif icin 'berbat', 'kotu', 'iade'; notr icin 'vardir', 'tarafindan' gibi nesnel kelimeler. Bu kelimeler domain bilgisiyle tamamen tutarli."

---

## SLAYT 9 - Kod Yapisi (6:15 - 7:00) [45 sn]

> "Koda kisa bir goz atalim. Solda IntentClassifierAgent: Pydantic BaseModel ile yapilandirilmis cikti uretiyor - language, domain, dataset, confidence, reasoning. Gemini ile structured JSON cikti aliyor, basarisizsa statik kurallara dusuyor."

> "Sagda train pipeline: her veri seti icin ayri bir YAML config dosyasi var. configs/ klasorunde TF-IDF parametreleri, model hiperparametreleri, egitim/test ayarlari tanimli. train_experiment.py bu YAML'i okuyup sirayla tum modelleri egitiyor."

> "Proje yapisi olarak: src/agents/ 3 ajan, src/models/ 8 model sinifi, src/preprocessing/ on isleme, app/ 6 sayfalik Streamlit arayuzu, configs/ YAML konfigurasyon dosyalari."

---

## SLAYT 10 - Model Karsilastirma Sonuclari (7:00 - 8:00) [1 dk]

> "Ana sonuclara gelelim. Turkish Sentiment veri setinde 8 modelin karsilastirmasi:"

> "Transformer yani BERTurk acik ara birinci: 0.930 Macro-F1. En iyi klasik model SVM: 0.877. Aralarinda 5.3 puanlik fark var."

> "Siralamaya bakarsak: Transformer, SVM, Naive Bayes, Logistic Regression, Random Forest, KNN, XGBoost, Decision Tree."

> "Ilginc bir gozlem: XGBoost bu veri setinde iyi performans gosteremiyor cunku TF-IDF 30 bin boyutlu sparse veri - tree-based modeller bunu verimli kullanamadigi icin 8 model icerisinde 7. sirada."

> "Decision Tree ise en kotu, 0.706 F1. Cunku tek bir agac 30 bin feature'dan anlamli bolunmeler yapamadi."

---

## SLAYT 11 - Confusion Matrix ve ROC (8:00 - 8:30) [30 sn]

> "Soldaki confusion matrix'lere bakarsak en belirgin fark negatif sinifta: Transformer 945 dogru negatif yakalarken, Decision Tree sadece 433. Negatif sinifin az verisiyle ogrenmek zor."

> "ROC egrileri de bunu destekliyor - Transformer'in AUC'si 0.991, Decision Tree'nin 0.828."

---

## SLAYT 12 - Negatif Sinif Analizi (8:30 - 9:00) [30 sn]

> "Negatif sinifa odaklanalim cunku en zor sinif bu. Transformer recall'u yuzde 81.5, SVM yuzde 65.3, Decision Tree yuzde 37.4."

> "Yani Decision Tree negatif yorumlarin ucte ikisini kaciriverdi. Bu sinif dengesizliginin etkisini cok net gosteriyor."

---

## SLAYT 13 - 10-Run Istatistiksel Analiz (9:00 - 9:45) [45 sn]

> "Sonuclarin istatistiksel anlamliligi icin 10 farkli random seed ile calistirdim. SVM ortalama 0.821 F1 ile tum klasik modelleri yeniyor. Tum p-degerleri 0.0001'in altinda - yani fark rastlantisal degil, istatistiksel olarak anlamli."

> "SVM neden en iyi klasik model? Cunku maximum-margin yaklasimi sparse TF-IDF uzayinda cok iyi calisiyor. 30 bin boyutta lineer ayrilabilirlik yuksek. L2 normalizasyon margin'i stabil tutuyor. Ve Naive Bayes'in aksine feature bagimsizligi varsaymiyor."

---

## SLAYT 14 - Capraz Veri Seti Analizi (9:45 - 10:15) [30 sn]

> "4 veri setinin hepsinde transformer birinci, SVM en iyi klasik model. Ama onemli bir bulgu: sinif dengesizligi arttikca transformer'in avantaji buyuyor. Dengeli veri setlerinde fark 1.2-1.9 puan, ama dengesiz Turkish Sentiment'te 5.3 puan."

> "Bu bize soyluyor ki: eger veriniz dengeliyse SVM ile cok yakin sonuc alabilirsiniz, ama dengesizlik varsa transformer'a ihtiyaciniz var."

---

## SLAYT 15 - Sonuc ve Gelecek (10:15 - 10:45) [30 sn]

> "Ozetlersek: SVM sparse TF-IDF'te en iyi klasik model, BERTurk her yerde birinci, sinif dengesizligi Macro-F1'i kritik yapiyor, K-Means basarisiz - supervised gerekli, 3 ajanli modular mimari farkli gorevlere esneklik sagliyor."

> "Limitasyonlar: SMOTE veya class-weighted loss uygulamadim, hiperparametre grid search yapmadim, XAI sadakat metrikleri degerlendirilmedi."

> "Simdi canli demo ile sistemi gostereyim."

---

## CANLI DEMO (10:45 - 15:00) [~4 dk]

### Hazirik
- Terminal'de `streamlit run app/Home.py` calistir
- Browser'da acik olsun

### Demo Akisi

#### 1. Home Sayfasi (10:45 - 11:15) [30 sn]

> "Bu Streamlit arayuzu. 6 sayfamiz var. Home sayfasinda projenin genel tanitimi ve mimari diyagrami var."

- Soldaki sidebar'i goster, sayfa listesini belirt

#### 2. Classify Text Sayfasi (11:15 - 12:45) [1.5 dk]

> "En onemli sayfa burasi - tek metin siniflandirma. Bir Turkce metin girelim:"

- Ornek metin yaz: **"Bu urun gercekten harika, kesinlikle tavsiye ederim! Kargo cok hizli geldi."**
- Experiment sec (turkish_sentiment iceren)
- Classify'a bas

> "Bakini, 3 ajan sirayla calisti:"
> - "Ajan 1 dili Turkce, domain'i sentiment olarak tespit etti"
> - "Ajan 2 secilen modelle siniflandirma yapti - tahmin pozitif, guven skoru yuksek"
> - "Ajan 3 aciklamalari uretdi"

- LIME grafigini goster:
> "LIME burada kelimelerin tahmini nasil etkiledigini gosteriyor. Yesil kelimeler pozitif tahmine destek veriyor - 'harika', 'tavsiye'. Kirmizi kelimeler zit yonde."

- SHAP grafigini goster:
> "SHAP da benzer ama matematiksel olarak farkli: Shapley degerleri ile her kelimenin katkisini hesapliyor."

- LLM aciklamasini goster:
> "Ve Gemini bunlari dogal dilde ozetliyor."

#### 3. Model Comparison Sayfasi (12:45 - 13:30) [45 sn]

> "Burada ayni metni 8 modelin hepsiyle ayni anda siniflandirabiliyoruz."

- Ayni Turkce metni gir, tum modelleri sec
- Sonuclari goster

> "Gordugunuz gibi tum modeller pozitif diyor ama guven skorlari farkli. SVM ve Transformer yuksek guvenle, Decision Tree daha dusuk guvenle tahmin ediyor. Bu tabloda accuracy, F1, precision, recall hepsini yan yana gorebiliyorsunuz."

#### 4. Dataset Explorer Sayfasi (13:30 - 14:00) [30 sn]

> "Burada veri setlerini gezebiliyoruz. Turkish Sentiment'i secersek ornekleri gorebiliriz - sinif dagilimi, metin uzunluklari, filtrele."

- Turkish Sentiment sec, birkac ornek goster

#### 5. Experiment Details Sayfasi (14:00 - 14:30) [30 sn]

> "Bu sayfada egitim sonuclarini detayli inceleyebiliyoruz. Confusion matrix, ROC egrileri, model bazinda metrikler, egitim sureleri - hepsi burada."

- Bir experiment sec, grafikleri goster

#### 6. Kapais (14:30 - 15:00) [30 sn]

> "Son olarak Train Models sayfasindan da bahsedeyim - buradan direkt arayuz uzerinden yeni model egitebiliyorsunuz. Dataset, model ve hiperparametreleri secip baslatiyorsunuz."

> "Ozetlemek gerekirse: 3 ajanli modular bir mimari, 8 model, 4 veri seti, LIME ve SHAP ile aciklanabilirlik, ve gordugunuz gibi tam islevsel bir web arayuzu. Tesekkurler, sorulariniz varsa alabilirim."

---

## OLASI SORULAR VE CEVAPLAR

**S: Neden Gemini sectin, OpenAI degil de?**
> "Gemini'nin structured output ozelligi var - JSON schema veriyorsunuz, ciktisi garanti o formatta geliyor. Bu Intent Classifier icin cok uygun. Ayrica ucretsiz API kotasi yeterli."

**S: SMOTE neden uygulamadin?**
> "TF-IDF 30 bin boyutlu sparse veri uzerine SMOTE uygulamak sentetik ornekleri meaningful olmayan bolgelere koyabilir. Gelecekte class-weighted SVM daha mantikli bir yaklasim olur."

**S: SVM neden XGBoost'u yeniyor?**
> "XGBoost tree-based bir model ve 30 bin boyutlu sparse verida her split icin cok fazla feature taramasi gerekiyor. SVM ise sparse veriyi direkt kernel trick ile isleme aliyor, daha verimli. Ayrica XGBoost'un hiperparametreleri (max_depth, learning_rate) tune edilmedi - varsayilan degerlerle calistirildi."

**S: Transformer neden bu kadar iyi?**
> "BERTurk 32 bin ornek uzerinde pre-train edilmis, Turkce morfolojiyi anlayan bir model. TF-IDF bag-of-words yaklasimi kelime sirasini kaybediyor, ama BERT attention mekanizmasiyla context'i yakalayabiliyor. Ozellikle negatif sinifta buyuk fark yaratiyor cunku az veriyle bile transfer learning'den faydalaniyor."

**S: LIME ve SHAP arasindaki fark ne?**
> "LIME lokal bir yaklasim - sadece o ornegi perturb ederek yakin bir bolge olusturuyor ve basit bir lineer model fitiyor. SHAP ise Shapley degerlerini kullaniyor, oyun teorisinden geliyor, her feature'in marginal katkisini hesapliyor. Matematiksel olarak SHAP daha saglam ama daha yavas."

**S: Neden 3 ajan? Tek bir model yetmez mi?**
> "Modular mimari avantaji: Ajan 1'i degistirmeden Ajan 2'ye yeni model ekleyebilirsiniz. Ajan 3 model-agnostik, hangi model olursa olsun aciklama uretebilir. Ayrica her ajan bagimsiz test edilebilir."

**S: Kodda config-driven yaklasimin avantaji ne?**
> "YAML config dosyalariyla ayni kodu farkli veri setlerinde, farkli parametrelerle calistirabiliyorum. Yeni bir veri seti eklemek icin sadece yeni bir YAML dosyasi yazmak yetiyor, kod degisikligi gerekmiyor."
