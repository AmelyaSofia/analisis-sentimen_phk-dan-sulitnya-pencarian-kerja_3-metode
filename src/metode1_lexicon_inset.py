import os
import sys
import urllib.request
import pandas as pd

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
INSET_DIR = os.path.join(BASE_DIR, "dictionaries", "inset")

POS_PATH = os.path.join(INSET_DIR, "positive.tsv")
NEG_PATH = os.path.join(INSET_DIR, "negative.tsv")

POS_URL = "https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv"
NEG_URL = "https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv"

def download_inset_if_needed():
    os.makedirs(INSET_DIR, exist_ok=True)
    for url, file_path in [(POS_URL, POS_PATH), (NEG_URL, NEG_PATH)]:
        if not os.path.exists(file_path):
            filename = os.path.basename(file_path)
            print(f"[*] Mengunduh {filename} dari GitHub...")
            try:
                urllib.request.urlretrieve(url, file_path)
                print(f"[✓] {filename} berhasil diunduh.")
            except Exception as e:
                print(f"[!] Gagal mengunduh {filename}: {e}")

def load_inset_lexicon():
    download_inset_if_needed()
    pos_dict = {}
    neg_dict = {}
    
    if os.path.exists(POS_PATH):
        df_pos = pd.read_csv(POS_PATH, sep='\t')
        for _, row in df_pos.iterrows():
            pos_dict[str(row['word']).strip().lower()] = float(row['weight'])
            
    if os.path.exists(NEG_PATH):
        df_neg = pd.read_csv(NEG_PATH, sep='\t')
        for _, row in df_neg.iterrows():
            neg_dict[str(row['word']).strip().lower()] = float(row['weight'])
            
    print(f"[*] Kamus InSet siap: {len(pos_dict)} kata positif, {len(neg_dict)} kata negatif.")
    return pos_dict, neg_dict

NEGATION_WORDS = {'tidak', 'bukan', 'jangan', 'belum', 'tanpa', 'kurang', 'tiada', 'jarang'}

def score_sentiment_inset(text, pos_dict, neg_dict):
    if not isinstance(text, str) or not text.strip():
        return 0.0, "Netral"
        
    words = text.split()
    total_score = 0.0
    i = 0
    
    while i < len(words):
        word = words[i]
        if word in NEGATION_WORDS and i + 1 < len(words):
            next_word = words[i + 1]
            if next_word in pos_dict:
                total_score -= pos_dict[next_word]
                i += 2
                continue
            elif next_word in neg_dict:
                total_score += abs(neg_dict[next_word])
                i += 2
                continue
                
        if word in pos_dict:
            total_score += pos_dict[word]
        elif word in neg_dict:
            total_score += neg_dict[word]
            
        i += 1
        
    if total_score > 0:
        label = "Positif"
    elif total_score < 0:
        label = "Negatif"
    else:
        label = "Netral"
        
    return total_score, label

def run_metode1_inset(input_file="hasil_preprocessing.csv", output_file="hasil_pelabelan_inset.csv"):
    input_path = os.path.join(DATA_PROCESSED, input_file)
    output_path = os.path.join(DATA_PROCESSED, output_file)
    
    if not os.path.exists(input_path):
        print(f"[!] File {input_path} tidak ditemukan! Jalankan preprocessing terlebih dahulu.")
        return None
        
    df = pd.read_csv(input_path)
    pos_dict, neg_dict = load_inset_lexicon()
    
    print(f"[*] Melakukan scoring sentimen pada {len(df)} komentar...")
    scores = []
    labels = []
    
    for text in df['clean_comment']:
        score, label = score_sentiment_inset(text, pos_dict, neg_dict)
        scores.append(score)
        labels.append(label)
        
    df['inset_score'] = scores
    df['sentiment'] = labels
    
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"[✓] Hasil pelabelan tersimpan di: {output_path}")
    
    distribusi = df['sentiment'].value_counts()
    persentase = df['sentiment'].value_counts(normalize=True) * 100
    
    print("\n" + "="*50)
    print("DISTRIBUSI SENTIMEN MASYARAKAT (METODE 1: INSET LEXICON)")
    print("="*50)
    for sent in ['Negatif', 'Positif', 'Netral']:
        count = distribusi.get(sent, 0)
        pct = persentase.get(sent, 0.0)
        print(f"{sent:<10} : {count:>4} komentar ({pct:.1f}%)")
    print("="*50)
    return df

def evaluate_inset(ground_truth_file="hasil_ground_truth.csv"):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
    gt_path = os.path.join(DATA_PROCESSED, ground_truth_file)
    if not os.path.exists(gt_path):
        return None
            
    df = pd.read_csv(gt_path)
    pos_dict, neg_dict = load_inset_lexicon()
    
    y_true = df['sentiment'].tolist()
    y_pred = []
    
    for text in df['clean_comment']:
        _, label = score_sentiment_inset(text, pos_dict, neg_dict)
        y_pred.append(label)
        
    labels_order = ['Negatif', 'Netral', 'Positif']
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    report_dict = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    
    results = {
        'accuracy': acc,
        'confusion_matrix': cm.tolist(),
        'labels': labels_order,
        'classification_report': report_dict,
        'total_count': len(df)
    }
    return results

if __name__ == "__main__":
    run_metode1_inset()
    eval_res = evaluate_inset()
    if eval_res:
        print(f"\nAkurasi InSet terhadap Ground Truth: {eval_res['accuracy']*100:.2f}%")
