import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
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

def train_and_evaluate_svm(input_file="hasil_ground_truth.csv"):
    input_path = os.path.join(DATA_PROCESSED, input_file)
    if not os.path.exists(input_path):
        print(f"[!] File {input_path} tidak ditemukan! Pastikan file data berlabel tersedia.")
        return None
        
    df = pd.read_csv(input_path)
    df = df.dropna(subset=['clean_comment', 'sentiment'])
    
    X = df['clean_comment']
    y = df['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.20, 
        random_state=42, 
        stratify=y
    )
    
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=1500)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    model_svm = SVC(kernel='linear', C=1.0, random_state=42)
    model_svm.fit(X_train_tfidf, y_train)
    
    y_pred = model_svm.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    
    labels_order = ['Negatif', 'Netral', 'Positif']
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)
    report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    
    results = {
        'model': model_svm,
        'tfidf': tfidf,
        'accuracy': acc,
        'confusion_matrix': cm.tolist(),
        'labels': labels_order,
        'classification_report': report_dict,
        'train_count': len(X_train),
        'test_count': len(X_test),
        'total_features': len(tfidf.get_feature_names_out())
    }
    
    return results

def run_metode2_svm():
    results = train_and_evaluate_svm()
    if not results:
        return
        
    print("\n" + "="*58)
    print("HASIL EVALUASI MODEL SVM (METODE 2)")
    print("="*58)
    print(f"Akurasi Pengujian: {results['accuracy'] * 100:.2f}%\n")
    print(f"Jumlah Fitur: {results['total_features']}")
    print(f"Confusion Matrix:\n{results['confusion_matrix']}")
    print("="*58)

if __name__ == "__main__":
    run_metode2_svm()
