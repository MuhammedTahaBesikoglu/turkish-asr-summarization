# Proje: Türkçe Konuşmadan Metne + Özetleme Sistemi

## Projenin amacı
- Türkçe ses kayıtlarını (toplantı, podcast, ders kaydı) metne çeviren ve ardından bu metnin özetini çıkaran uçtan uca bir sistem.
- İki ana bileşen var:
  1. **ASR (Automatic Speech Recognition):** OpenAI Whisper modelinin Türkçe veriyle fine-tune edilmesi.
  2. **Özetleme:** ASR çıktısını bir dil modeliyle özetleyen ikinci aşama.
- Bu bir **portföy projesi**. GitHub'da yayınlanacak ve CV'de yer alacak. Kod kalitesi, tekrarlanabilirlik ve dokümantasyon en az modelin başarısı kadar önemli.

## Temel hedefler (başarı kriterleri)
- Hazır (zero-shot) Whisper modelinin Türkçe test setindeki WER'ini ölçmek: **baseline**.
- Fine-tune edilmiş modelin aynı test setindeki WER'ini ölçmek ve baseline ile karşılaştırmak.
- Fine-tune edilmiş modeli Hugging Face Hub'a model kartıyla birlikte yüklemek.
- Ses dosyası yükleyince transkript ve özet döndüren bir web demosu (Gradio) yapmak ve Hugging Face Spaces'te yayınlamak.
- Sonuçları tablo ve grafiklerle anlatan bir README hazırlamak.

## Önemli kurallar (Claude Code için)
- **Sahte, uydurma veya tahmini sonuç yazma.** README'deki, model kartındaki veya raporlardaki tüm WER/metrik değerleri gerçekten çalıştırılmış değerlendirme script'lerinin çıktısından gelmeli. Bir değer henüz ölçülmediyse "TBD" yaz.
- Bir kütüphane fonksiyonu, model adı veya veri seti yolu hakkında emin değilsen tahmin etme; dokümantasyonu kontrol et ya da bana sor.
- Büyük değişikliklerden önce plan sun, onayımı al.
- Her aşama bitince çalışır durumda olmalı ve küçük, anlamlı commit'lerle ilerle.

## Veri seti
- **Ana veri:** Mozilla Common Voice, Türkçe (`tr`).
- **DİKKAT:** Ekim 2025 itibarıyla Common Voice veri setleri yalnızca **Mozilla Data Collective** (https://datacollective.mozillafoundation.org) üzerinden dağıtılıyor. Hugging Face'teki `mozilla-foundation/common_voice_*` sayfaları artık veriyi `load_dataset` ile sunmuyor. Bu yüzden:
  - Veri Mozilla Data Collective'den hesap açılıp manuel indirilecek (bunu ben yapacağım).
  - Kod, indirilmiş yerel dosyalardan (`.tsv` metadata + ses klipleri) okuyacak şekilde yazılmalı.
  - İndirilen sürüm numarası README'de belirtilmeli (tekrarlanabilirlik için).
- Common Voice'un resmi train / dev / test bölünmeleri kullanılmalı. Test seti eğitimde **asla** kullanılmamalı.
- Veri `data/` klasöründe durur ve `.gitignore` ile repoya eklenmez.

## Model ve eğitim
- **Başlangıç modeli:** `openai/whisper-small` (donanım yeterliyse ileride `whisper-medium` denenebilir).
- **Kütüphaneler:** PyTorch, Hugging Face `transformers`, `datasets`, `evaluate`, `accelerate`.
- **Verimli eğitim:** `peft` ile LoRA. Hem tam fine-tune hem LoRA denenip sonuçları (WER, eğitim süresi, bellek kullanımı) karşılaştırılabilir. Bu karşılaştırma projenin öne çıkan analizlerinden biri olacak.
- Ses 16 kHz'e yeniden örneklenmeli (Whisper'ın beklediği format).
- Mixed precision (fp16) ve gerekirse gradient accumulation kullanılmalı.
- Referans: Hugging Face'in "Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers" blog yazısı.
- **Donanım:**
  - **Eğitim:** Google Colab, T4 GPU (16 GB VRAM).
  - **Geliştirme / test / küçük değerlendirmeler:** Yerel Windows 11, NVIDIA RTX 3050 Ti Laptop (4 GB VRAM). Yerelde tam fine-tune yapılmaz; kod hem yerelde hem Colab'da çalışacak şekilde yazılmalı (yollar config'den gelmeli).
  - Yerel Python ortamı: 3.12 (`.venv`).

## Değerlendirme
- **Metrik:** WER (Word Error Rate), ek olarak CER (Character Error Rate). `jiwer` veya `evaluate` kullanılabilir.
- **Türkçe metin normalizasyonu çok önemli:**
  - Python'un varsayılan `.lower()` fonksiyonu Türkçe "I/ı" ve "İ/i" harflerini yanlış dönüştürür. Türkçeye özel küçük harf dönüşümü yazılmalı.
  - Noktalama işaretlerinin temizlenmesi, sayıların yazımı gibi normalizasyon kuralları tek bir fonksiyonda toplanmalı ve hem baseline hem fine-tune değerlendirmesinde **aynı** fonksiyon kullanılmalı.
  - WER hem normalize edilmiş hem ham metin üzerinden raporlanabilir.
- **Hata analizi:** En çok hata yapılan kelimeler, uzun/kısa kliplerde performans farkı gibi analizler bir notebook'ta yapılmalı.

## Özetleme aşaması
- ASR çıktısı bir özetleme modelinden geçirilecek.
- Model seçimi henüz kesinleşmedi. Adaylar:
  - Hugging Face Hub'da Türkçe özetleme için fine-tune edilmiş açık kaynak bir model
  - Çok dilli, açık ağırlıklı bir instruction modeli
- Claude Code: model önermeden önce Hub'da gerçekten var olduğunu, lisansını ve Türkçe desteğini kontrol et.
- Uzun transkriptler için parçalara bölme (chunking) ve parça özetlerini birleştirme stratejisi gerekecek.

## Demo ve deploy
- **Gradio** arayüzü: ses dosyası yükleme veya mikrofonla kayıt → transkript → özet.
- Hugging Face Spaces'te yayınlanacak.
- (Opsiyonel, ileri aşama) FastAPI ile REST API + Docker ile paketleme.

## Deney takibi
- Weights & Biases veya MLflow ile tüm eğitim deneyleri (hiperparametreler, loss, WER) kayıt altına alınmalı.
- Hiperparametreler kod içine gömülmemeli; `configs/` altındaki YAML dosyalarından okunmalı.

## Önerilen repo yapısı
```
turkish-asr-summarization/
├── configs/              # Eğitim ve değerlendirme YAML dosyaları
├── data/                 # Ham ve işlenmiş veri (gitignore)
├── notebooks/            # Keşifsel veri analizi ve hata analizi
├── src/
│   ├── data/             # Veri yükleme, ön işleme, normalizasyon
│   ├── training/         # Fine-tune script'leri (full + LoRA)
│   ├── evaluation/       # WER/CER hesaplama, baseline değerlendirme
│   ├── summarization/    # Özetleme hattı
│   └── pipeline.py       # Ses → transkript → özet uçtan uca akış
├── app/                  # Gradio demo
├── tests/                # pytest testleri (özellikle normalizasyon fonksiyonu)
├── results/              # Metrik tabloları ve grafikler
├── requirements.txt
├── README.md
└── CLAUDE.md
```

## Kod standartları 
- Python 3.10+
- Type hint kullanımı, kısa ve anlaşılır docstring'ler
- `ruff` ile lint/format
- Notebook'lar sadece analiz içindir; asıl mantık `src/` altında olmalı
- Türkçe normalizasyon fonksiyonu için mutlaka unit test yazılmalı

## Yol haritası (aşamalar)
1. **Kurulum:** Repo yapısı, `requirements.txt`, ortam kurulumu, `.gitignore`
2. **Veri:** Common Voice Türkçe verisini yerelden okuma, keşifsel analiz (klip süreleri, cümle uzunlukları, konuşmacı sayısı)
3. **Normalizasyon:** Türkçe metin normalizasyon fonksiyonu + testleri
4. **Baseline:** Hazır `whisper-small` ile test setinde WER/CER ölçümü
5. **Fine-tune:** Tam fine-tune ve LoRA eğitimleri, deney takibi
6. **Değerlendirme:** Karşılaştırma tablosu + hata analizi notebook'u
7. **Özetleme:** Model seçimi ve özetleme hattı
8. **Pipeline:** Uçtan uca `ses → transkript → özet`
9. **Demo:** Gradio arayüzü ve Hugging Face Spaces deploy
10. **Yayın:** Modeli Hub'a model kartıyla yükleme, README'yi gerçek sonuçlarla tamamlama
11. **(Opsiyonel)** FastAPI + Docker

## README'de olması gerekenler
- Projenin kısa açıklaması ve demo linki
- Mimari diyagramı (ses → Whisper → metin → özetleyici → özet)
- Baseline ve fine-tune sonuç tablosu (gerçek değerlerle)
- Full fine-tune ve LoRA karşılaştırması
- Kurulum ve çalıştırma adımları
- Kullanılan veri setinin sürümü ve lisansı
- Bilinen sınırlamalar