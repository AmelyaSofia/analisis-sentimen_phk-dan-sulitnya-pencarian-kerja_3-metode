# Analisis Sentimen: Fenomena PHK dan Lapangan Kerja di Indonesia 🇮🇩

Proyek tugas analisis sentimen opini publik terhadap isu Pemutusan Hubungan Kerja (PHK) dan ketersediaan lapangan kerja di Indonesia menggunakan data komentar YouTube multi-video dengan komparasi **3 Metode**:
1. **InSet Lexicon** (Rule-based Dictionary Scoring)
2. **Support Vector Machine / SVM** (Supervised Machine Learning)
3. **TF-IDF + Cosine Similarity** (Vector Space Model Centroid Classification)

---

## 📁 Struktur Direktori Proyek

```text
tugas1/
│
├── app.py                            # Aplikasi Web Flask (Dashboard Komparasi 3 Metode)
├── README.md                         # Dokumentasi proyek
│
├── data/
│   ├── raw/
│   │   └── hasil_scraper.csv                 # Data mentah komentar YouTube (300 data)
│   └── processed/
│       ├── hasil_preprocessing.csv           # Data hasil cleaning & normalisasi (296 data)
│       ├── hasil_ground_truth.csv            # Dataset Ground Truth Acuan (296 data)
│       ├── ground_truth_acuan.xlsx           # Berkas Excel Ground Truth Acuan (murni)
│       └── hasil_evaluasi_3_metode.xlsx      # Berkas Excel Matriks Evaluasi 3 Metode
│
├── dictionaries/
│   └── inset/
│       ├── positive.tsv                      # 3.607 kata positif InSet
│       └── negative.tsv                      # 6.606 kata negatif InSet
│
├── src/
│   ├── scraper.py                            # Modul scraping YouTube Data API v3
│   ├── preprocessing.py                      # Modul pembersihan teks & normalisasi slang
│   ├── metode1_lexicon_inset.py              # Modul analisis sentimen InSet Lexicon
│   ├── metode2_svm.py                        # Modul klasifikasi SVM & evaluasi metrik
│   └── metode3_cosine_similarity.py          # Modul TF-IDF + Cosine Similarity Centroid
│
└── templates/
    ├── index.html                            # Antarmuka Dashboard Web (Tailwind CSS + Chart.js)
    └── viewer.html                           # Antarmuka Viewer Berkas Excel (.xlsx) di Browser
```

---

## 🚀 Cara Menjalankan

### Menjalankan Dashboard Web Komparatif
Cukup jalankan satu perintah berikut di terminal:
```powershell
python D:\ansos\tugas1\app.py
```
Lalu buka browser Anda di: **`http://127.0.0.1:5000`**

Di web dashboard ini Anda bisa:
* Melihat perbandingan komparatif ketiga metode secara berdampingan.
* Menggunakan fitur **Live Sentiment Tester** untuk menguji teks komentar baru sekaligus ke 3 metode secara *real-time*.
* Melihat matriks evaluasi dan *Confusion Matrix* untuk ketiga metode (InSet, SVM, dan Cosine Similarity) terhadap Ground Truth Acuan.
* Membuka langsung di browser atau mengunduh berkas Microsoft Excel (.xlsx):
  * `ground_truth_acuan.xlsx` (seluruh 296 data komentar asli, hasil preprocessing, dan label Ground Truth Acuan murni).
  * `hasil_evaluasi_3_metode.xlsx` (matriks kuantitatif Precision, Recall, F1-Score, dan Akurasi komparatif ketiga metode).
* Mengeksplorasi seluruh 296 data komentar scraped dengan fitur pencarian instan dan perbandingan hasil ketiga metode (InSet, SVM, Cosine Similarity) terhadap Ground Truth Acuan.

---

## 📊 Hasil Evaluasi Komparasi 3 Metode (Terhadap Ground Truth Acuan)

> **Catatan Metodologi:** Acuan evaluasi (*Ground Truth Acuan*) dibuat melalui anotasi terhadap 296 data komentar dengan mempertimbangkan konteks isu PHK, sarkasme, dan satir politik. InSet Lexicon tidak lagi dijadikan acuan sirkular, melainkan diuji sebagai metode independen.

| Aspek Evaluasi | Metode 1: InSet Lexicon | Metode 2: SVM (Linear) | Metode 3: TF-IDF + Cosine Sim |
| :--- | :--- | :--- | :--- |
| **Paradigma** | Rule-based / Kamus kata | Supervised Machine Learning | Vector Space Model (Centroid) |
| **Fitur Teks** | Pencocokan Kata + Negasi | TF-IDF (Unigram + Bigram) | TF-IDF (Centroid Vector) |
| **Kebutuhan Training** | Tidak butuh data latih | Butuh Data Latih (Train 80%) | Butuh Data Latih (Centroid) |
| **Akurasi Pengujian** | **42.23%** (Rentan sarkasme) | **75.00%** (Tertinggi) | **70.00%** |
| **Precision (Weighted)** | **0.64** | **0.68** | **0.65** |
| **Recall (Weighted)** | **0.42** | **0.75** (Tertinggi) | **0.70** |
| **F1-Score (Weighted)** | **0.45** | **0.68** (Tertinggi) | **0.68** |
| **F1-Score Negatif** | **0.50** | **0.85** (Tertinggi) | **0.83** |
