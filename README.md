Sentiment Analysis — Emotion Classification
Task 1: Classical ML Approach
Bu layihə mətn daxilindən 6 emosiyanı klassifikasiya edən tam bir Sentiment Analysis pipeline-ı tətbiq edir. Klassik Machine Learning yanaşması əsasında qurulmuşdur: TF-IDF Vectorization + Logistic Regression.

📁 Fayl Strukturu
TASK 1/
├── Sentiment_Analysis.ipynb   # Əsas notebook — bütün pipeline buradadır
├── explore_data.py            # Dataset-ə ilkin baxış skripti
├── training.csv               # Öyrətmə datası (16,000 nümunə)
├── test.csv                   # Test datası (2,000 nümunə)
├── validation.csv             # Validasiya datası (2,000 nümunə)
└── Read me 2 - Emosiyalar.txt # Emosiya sinifləri haqqında qısa qeyd

🎯 Emosiya Sinifləri
LabelEmosiya0Sadness1Anger2Love3Fear4Joy5Surprise

📊 Dataset
SplitNümunə sayıTraining16,000Test2,000Validation2,000Cəmi20,000
Hər sətirdə text (ingilis dilində cümlə) və label (0–5 arası emosiya kodu) sütunları var.

⚙️ Texnologiyalar

Python 3
pandas, numpy — data emalı
scikit-learn — TF-IDF, Logistic Regression, metrics
nltk — stopword siyahısı
matplotlib, seaborn — vizualizasiya
gradio — interaktiv interfeys


🔄 Pipeline
Xam Mətn
   │
   ▼
Preprocessing
  • Kiçik hərfə çevirmə
  • URL, e-mail silmə
  • Rəqəm silmə
  • Durğu işarəsi silmə
  • Stopword silmə (NLTK)
   │
   ▼
TF-IDF Vectorization
  • max_features=5000
  • ngram_range=(1, 2)
  • min_df=2, max_df=0.8
   │
   ▼
Logistic Regression
  • solver='lbfgs'
  • max_iter=1000
   │
   ▼
Emosiya Proqnoz (0–5)

🚀 İstifadə
1. Asılılıqları quraşdırın
bashpip install pandas numpy scikit-learn nltk matplotlib seaborn gradio
2. Notebook-u çalışdırın
bashjupyter notebook Sentiment_Analysis.ipynb
Hücrələri ardıcıl icra edin. Sonuncu hücrədə Gradio interfeysi başlayacaq.
3. Komand xəttindən data baxışı
bashpython explore_data.py

🖥️ Gradio İnterfeysi
Notebook tam icra edildikdən sonra real-time emosiya analizi üçün veb interfeys açılır. İstifadəçi cümlə daxil edir, model hər 6 emosiya üçün ehtimal faizini qaytarır.
pythoniface.launch()          # lokal
iface.launch(share=True) # ictimai link (paylaşmaq üçün)

📈 Qiymətləndirmə Metrikaları
Modelin performansı aşağıdakı göstəricilər vasitəsilə ölçülür:

Accuracy — train / test / validation üzrə
Classification Report — precision, recall, F1 hər sinif üçün
Confusion Matrix — hansı emosiyaların bir-biri ilə qarışdırıldığını göstərir
Misclassification Analysis — ən çox həll edilməsi çətin olan sinif cütləri


📝 Qeydlər

Model ingilis dilindəki mətnlər üçün hazırlanmışdır.
Preprocessing-də stopword-lər standart NLTK ingilis siyahısına əsasən silinir.
Daha yüksək performans üçün max_features artırıla, ya da model SVM / Random Forest ilə əvəz edilə bilər.
