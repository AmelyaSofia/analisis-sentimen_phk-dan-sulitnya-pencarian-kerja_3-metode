import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")

sys.path.append(os.path.join(BASE_DIR, "src"))
try:
    from preprocessing import preprocess_comment
except ImportError:
    preprocess_comment = lambda text: (text.lower(), text.lower().split())

def train_and_evaluate_cosine_sim(input_file="hasil_ground_truth.csv"):
    input_path = os.path.join(DATA_PROCESSED, input_file)
    if not os.path.exists(input_path):
        print(f"[!] File {input_path} tidak ditemukan! Pastikan file data berlabel tersedia.")
        return None
        
    df = pd.read_csv(input_path)
    df = df.dropna(subset=['clean_comment', 'sentiment'])
    
    X = df['clean_comment']
    y = df['sentiment']
    
    # Split data persis sama dengan SVM (80% train, 20% test, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.20, 
        random_state=42, 
        stratify=y
    )
    
    # 1. Ekstraksi Fitur TF-IDF
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=1500)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    classes = ['Negatif', 'Netral', 'Positif']
    centroids = {}
    y_train_arr = y_train.values
    
    # 2. Pembentukan Centroid Profil Sentimen
    for cls in classes:
        idx = np.where(y_train_arr == cls)[0]
        if len(idx) > 0:
            # Rata-rata bobot TF-IDF untuk dokumen di kelas tsb
            mean_vec = np.asarray(X_train_tfidf[idx].mean(axis=0))
            # Normalisasi L2 vektor centroid
            norm = np.linalg.norm(mean_vec)
            if norm > 0:
                mean_vec = mean_vec / norm
            centroids[cls] = mean_vec
        else:
            centroids[cls] = np.zeros((1, X_train_tfidf.shape[1]))
            
    # 3. Prediksi Data Uji dengan Cosine Similarity
    y_pred = []
    test_similarities = []
    
    for i in range(X_test_tfidf.shape[0]):
        sample_vec = X_test_tfidf[i].toarray()
        sim_scores = {}
        for cls in classes:
            sim = float(cosine_similarity(sample_vec, centroids[cls])[0][0])
            sim_scores[cls] = sim
            
        best_cls = max(sim_scores, key=sim_scores.get)
        y_pred.append(best_cls)
        test_similarities.append(sim_scores)
        
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    
    results = {
        'tfidf': tfidf,
        'centroids': centroids,
        'classes': classes,
        'accuracy': acc,
        'confusion_matrix': cm.tolist(),
        'classification_report': report_dict,
        'train_count': len(X_train),
        'test_count': len(X_test),
        'total_features': len(tfidf.get_feature_names_out())
    }
    
    return results

def predict_cosine_similarity(clean_text, tfidf, centroids, classes):
    """
    Menghitung skor Cosine Similarity suatu teks baru terhadap masing-masing kelas sentimen.
    """
    if not clean_text.strip():
        return "Netral", {cls: 0.0 for cls in classes}
        
    vec = tfidf.transform([clean_text]).toarray()
    # Jika seluruh kata tidak ada di vocab (vektor nol), kembalikan Netral
    if np.linalg.norm(vec) == 0:
        return "Netral", {cls: 0.0 for cls in classes}
        
    scores = {}
    for cls in classes:
        sim = float(cosine_similarity(vec, centroids[cls])[0][0])
        scores[cls] = round(sim, 4)
        
    best_cls = max(scores, key=scores.get)
    return best_cls, scores

def run_metode3_cosine():
    print("[*] Melatih model TF-IDF + Cosine Similarity...")
    results = train_and_evaluate_cosine_sim()
    if not results:
        return
        
    print("\n" + "="*58)
    print("HASIL EVALUASI METODE 3: TF-IDF + COSINE SIMILARITY")
    print("="*58)
    print(f"Akurasi Pengujian: {results['accuracy'] * 100:.2f}%\n")
    print("Classification Report:")
    for cls in results['classes']:
        rep = results['classification_report'].get(cls, {})
        print(f"  - {cls:<8}: Precision={rep.get('precision', 0):.4f}, Recall={rep.get('recall', 0):.4f}, F1={rep.get('f1-score', 0):.4f}")
    print("="*58)
    
    # Pengujian contoh
    test_comment = "banyak phk dan pabrik bangkrut di mana mana"
    clean_txt, _ = preprocess_comment(test_comment)
    label, sims = predict_cosine_similarity(clean_txt, results['tfidf'], results['centroids'], results['classes'])
    print(f"\nContoh Uji: \"{test_comment}\"")
    print(f"Skor Cosine: {sims}")
    print(f"Prediksi: [{label}]")

if __name__ == "__main__":
    run_metode3_cosine()
