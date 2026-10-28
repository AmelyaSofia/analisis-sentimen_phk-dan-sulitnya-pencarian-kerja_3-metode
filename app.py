import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, jsonify, send_file, Response

# Konfigurasi path modul src
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
sys.path.append(SRC_DIR)

from preprocessing import preprocess_comment
from metode1_lexicon_inset import load_inset_lexicon, score_sentiment_inset, evaluate_inset
from metode2_svm import train_and_evaluate_svm
from metode3_cosine_similarity import train_and_evaluate_cosine_sim, predict_cosine_similarity

app = Flask(__name__)

# ==============================================================================
# INISIALISASI DATA & 3 METODE SAAT APLIKASI DIJALANKAN
# ==============================================================================
print("[*] Menginisialisasi kamus InSet, melatih model SVM & Cosine Similarity...")
pos_dict, neg_dict = load_inset_lexicon()
inset_eval_results = evaluate_inset()
svm_results = train_and_evaluate_svm()
cos_results = train_and_evaluate_cosine_sim()

# Baca data berlabel Ground Truth Acuan
data_path = os.path.join(DATA_PROCESSED, "hasil_ground_truth.csv")

if os.path.exists(data_path):
    df_data = pd.read_csv(data_path)
else:
    df_data = pd.DataFrame()

def generate_export_excel():
    """Menghasilkan berkas Excel untuk Ground Truth Acuan (murni tanpa prediksi) dan Hasil Evaluasi 3 Metode"""
    # 1. Dataset Ground Truth Acuan (hanya komentar dan label Ground Truth, tanpa prediksi metode)
    if not df_data.empty:
        df_copy = df_data.copy()
        gt_col = df_copy['sentiment'] if 'sentiment' in df_copy.columns else df_copy.get('ground_truth', 'Netral')
        
        df_export_gt = pd.DataFrame({
            'No': range(1, len(df_copy) + 1),
            'Username': df_copy['author'] if 'author' in df_copy.columns else '',
            'Komentar_Asli': df_copy['comment'],
            'Teks_Preprocessing': df_copy['clean_comment'],
            'Ground_Truth_Acuan': gt_col
        })
        gt_file = os.path.join(DATA_PROCESSED, "ground_truth_acuan.xlsx")
        df_export_gt.to_excel(gt_file, index=False, engine='openpyxl')

    # 2. Hasil Evaluasi Komparasi 3 Metode
    eval_rows = []
    methods_data = [
        ('Metode 1: InSet Lexicon', inset_eval_results),
        ('Metode 2: SVM (Linear)', svm_results),
        ('Metode 3: TF-IDF + Cosine Sim', cos_results)
    ]
    for m_name, res in methods_data:
        if res and 'classification_report' in res:
            rep = res['classification_report']
            acc = res.get('accuracy', 0.0)
            for cat in ['Negatif', 'Netral', 'Positif', 'macro avg', 'weighted avg']:
                if cat in rep:
                    eval_rows.append({
                        'Metode': m_name,
                        'Kategori_Evaluasi': cat,
                        'Precision': round(rep[cat].get('precision', 0), 4),
                        'Recall': round(rep[cat].get('recall', 0), 4),
                        'F1_Score': round(rep[cat].get('f1-score', 0), 4),
                        'Support': int(rep[cat].get('support', 0)),
                        'Akurasi_Metode': f"{acc * 100:.2f}%"
                    })
    df_export_eval = pd.DataFrame(eval_rows)
    eval_file = os.path.join(DATA_PROCESSED, "hasil_evaluasi_3_metode.xlsx")
    df_export_eval.to_excel(eval_file, index=False, engine='openpyxl')

generate_export_excel()

def format_classification_metrics(report_dict):
    if not report_dict:
        return {}
    res = {}
    for cls in ['Negatif', 'Netral', 'Positif']:
        if cls in report_dict:
            res[cls] = {
                'precision': f"{report_dict[cls].get('precision', 0):.2f}",
                'recall': f"{report_dict[cls].get('recall', 0):.2f}",
                'f1': f"{report_dict[cls].get('f1-score', 0):.2f}",
                'support': int(report_dict[cls].get('support', 0))
            }
        else:
            res[cls] = {'precision': '0.00', 'recall': '0.00', 'f1': '0.00', 'support': 0}
            
    w_avg = report_dict.get('weighted avg', {})
    res['weighted_avg'] = {
        'precision': f"{w_avg.get('precision', 0):.2f}",
        'recall': f"{w_avg.get('recall', 0):.2f}",
        'f1': f"{w_avg.get('f1-score', 0):.2f}",
        'support': int(w_avg.get('support', 0))
    }
    
    macro_avg = report_dict.get('macro avg', {})
    res['macro_avg'] = {
        'precision': f"{macro_avg.get('precision', 0):.2f}",
        'recall': f"{macro_avg.get('recall', 0):.2f}",
        'f1': f"{macro_avg.get('f1-score', 0):.2f}"
    }
    return res

@app.route('/')
def index():
    total_comments = len(df_data)
    distribusi = df_data['sentiment'].value_counts().to_dict() if not df_data.empty else {}
    
    pos_count = distribusi.get('Positif', 0)
    neg_count = distribusi.get('Negatif', 0)
    net_count = distribusi.get('Netral', 0)
    
    pos_pct = round((pos_count / total_comments * 100), 1) if total_comments > 0 else 0
    neg_pct = round((neg_count / total_comments * 100), 1) if total_comments > 0 else 0
    net_pct = round((net_count / total_comments * 100), 1) if total_comments > 0 else 0
    
    # 1. Metrik InSet Lexicon (Metode 1) terhadap Ground Truth
    inset_acc = round(inset_eval_results['accuracy'] * 100, 2) if inset_eval_results else 0
    inset_cm = inset_eval_results['confusion_matrix'] if inset_eval_results else []
    inset_report = inset_eval_results['classification_report'] if inset_eval_results else {}
    inset_metrics = format_classification_metrics(inset_report)

    # 2. Metrik SVM (Metode 2)
    svm_acc = round(svm_results['accuracy'] * 100, 2) if svm_results else 0
    svm_cm = svm_results['confusion_matrix'] if svm_results else []
    svm_report = svm_results['classification_report'] if svm_results else {}
    svm_metrics = format_classification_metrics(svm_report)
    
    # 3. Metrik Cosine Similarity (Metode 3)
    cos_acc = round(cos_results['accuracy'] * 100, 2) if cos_results else 0
    cos_cm = cos_results['confusion_matrix'] if cos_results else []
    cos_report = cos_results['classification_report'] if cos_results else {}
    cos_metrics = format_classification_metrics(cos_report)
    
    # 4. Prediksi ketiga metode (InSet, SVM, Cosine Similarity) untuk seluruh komentar
    table_data = []
    if not df_data.empty:
        df_copy = df_data.copy()
        
        # Metode 1: InSet Lexicon
        if 'inset_sentiment' not in df_copy.columns or 'inset_score' not in df_copy.columns:
            in_scores = []
            in_labels = []
            for t in df_copy['clean_comment'].fillna(''):
                sc, lb = score_sentiment_inset(t, pos_dict, neg_dict)
                in_scores.append(sc)
                in_labels.append(lb)
            df_copy['inset_score'] = in_scores
            df_copy['inset_sentiment'] = in_labels
            
        # Metode 2: SVM
        if svm_results and 'model' in svm_results and 'tfidf' in svm_results:
            svm_vecs = svm_results['tfidf'].transform(df_copy['clean_comment'].fillna(''))
            df_copy['svm_sentiment'] = svm_results['model'].predict(svm_vecs)
        else:
            df_copy['svm_sentiment'] = '-'
            
        # Metode 3: Cosine Similarity
        if cos_results and 'tfidf' in cos_results and 'centroids' in cos_results:
            cos_vecs = cos_results['tfidf'].transform(df_copy['clean_comment'].fillna(''))
            cos_preds = []
            for i in range(cos_vecs.shape[0]):
                r_vec = cos_vecs[i].toarray()
                if np.linalg.norm(r_vec) == 0:
                    cos_preds.append("Netral")
                else:
                    sim_dict = {
                        cls: float(cosine_similarity(r_vec, cos_results['centroids'][cls])[0][0])
                        for cls in cos_results['classes']
                    }
                    cos_preds.append(max(sim_dict, key=sim_dict.get))
            df_copy['cosine_sentiment'] = cos_preds
        else:
            df_copy['cosine_sentiment'] = '-'
            
        # Ground Truth Acuan
        df_copy['ground_truth'] = df_copy['sentiment'] if 'sentiment' in df_copy.columns else df_copy.get('ground_truth', 'Netral')
        
        cols = ['author', 'comment', 'clean_comment', 'inset_score', 'inset_sentiment', 'svm_sentiment', 'cosine_sentiment', 'ground_truth']
        table_data = df_copy[cols].to_dict(orient='records')
        
    return render_template(
        'index.html',
        total_comments=total_comments,
        pos_count=pos_count,
        neg_count=neg_count,
        net_count=net_count,
        pos_pct=pos_pct,
        neg_pct=neg_pct,
        net_pct=net_pct,
        inset_acc=inset_acc,
        inset_cm=inset_cm,
        inset_report=inset_report,
        inset_metrics=inset_metrics,
        svm_acc=svm_acc,
        svm_cm=svm_cm,
        svm_report=svm_report,
        svm_metrics=svm_metrics,
        svm_total_features=svm_results.get('total_features', 0) if svm_results else 0,
        cos_acc=cos_acc,
        cos_cm=cos_cm,
        cos_report=cos_report,
        cos_metrics=cos_metrics,
        table_data=table_data
    )

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() or {}
    raw_text = data.get('text', '').strip()
    
    if not raw_text:
        return jsonify({'error': 'Teks komentar tidak boleh kosong'}), 400
        
    # 1. Preprocessing
    clean_text, tokens = preprocess_comment(raw_text)
    
    # 2. Prediksi Metode 1 (InSet Lexicon)
    inset_score, inset_label = score_sentiment_inset(clean_text, pos_dict, neg_dict)
    matched_pos = [word for word in clean_text.split() if word in pos_dict]
    matched_neg = [word for word in clean_text.split() if word in neg_dict]
    
    # 3. Prediksi Metode 2 (SVM)
    svm_label = "N/A"
    if svm_results and 'model' in svm_results and 'tfidf' in svm_results:
        vec = svm_results['tfidf'].transform([clean_text])
        svm_label = str(svm_results['model'].predict(vec)[0])
        
    # 4. Prediksi Metode 3 (TF-IDF + Cosine Similarity)
    cos_label = "N/A"
    cos_scores = {}
    if cos_results and 'tfidf' in cos_results and 'centroids' in cos_results:
        cos_label, cos_scores = predict_cosine_similarity(
            clean_text, 
            cos_results['tfidf'], 
            cos_results['centroids'], 
            cos_results['classes']
        )
        
    return jsonify({
        'original_text': raw_text,
        'clean_text': clean_text,
        'tokens': tokens,
        'inset': {
            'score': inset_score,
            'label': inset_label,
            'matched_pos': matched_pos,
            'matched_neg': matched_neg
        },
        'svm': {
            'label': svm_label
        },
        'cosine': {
            'label': cos_label,
            'scores': cos_scores
        }
    })

# ==============================================================================
# ROUTE AKSES & UNDUH BERKAS EXCEL (.xlsx)
# ==============================================================================
@app.route('/download/ground-truth-excel')
@app.route('/download/ground-truth-csv')
def download_ground_truth_excel():
    file_path = os.path.join(DATA_PROCESSED, "ground_truth_acuan.xlsx")
    if not os.path.exists(file_path):
        generate_export_excel()
    return send_file(
        file_path, 
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 
        as_attachment=True, 
        download_name='ground_truth_acuan.xlsx'
    )

@app.route('/download/evaluasi-metode-excel')
@app.route('/download/evaluasi-metode-csv')
def download_evaluasi_metode_excel():
    file_path = os.path.join(DATA_PROCESSED, "hasil_evaluasi_3_metode.xlsx")
    if not os.path.exists(file_path):
        generate_export_excel()
    return send_file(
        file_path, 
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 
        as_attachment=True, 
        download_name='hasil_evaluasi_3_metode.xlsx'
    )

@app.route('/view/ground-truth-excel')
@app.route('/view/ground-truth-csv')
def view_ground_truth_excel():
    file_path = os.path.join(DATA_PROCESSED, "ground_truth_acuan.xlsx")
    if not os.path.exists(file_path):
        generate_export_excel()
    df = pd.read_excel(file_path)
    return render_template(
        'viewer.html',
        title='Ground Truth Acuan (.xlsx)',
        filename='ground_truth_acuan.xlsx',
        file_type='Ground Truth Acuan (Murni tanpa prediksi metode)',
        download_url='/download/ground-truth-excel',
        headers=df.columns.tolist(),
        rows=df.fillna('').values.tolist(),
        total_rows=len(df)
    )

@app.route('/view/evaluasi-metode-excel')
@app.route('/view/evaluasi-metode-csv')
def view_evaluasi_metode_excel():
    file_path = os.path.join(DATA_PROCESSED, "hasil_evaluasi_3_metode.xlsx")
    if not os.path.exists(file_path):
        generate_export_excel()
    df = pd.read_excel(file_path)
    return render_template(
        'viewer.html',
        title='Hasil Evaluasi 3 Metode (.xlsx)',
        filename='hasil_evaluasi_3_metode.xlsx',
        file_type='Matriks Evaluasi Kuantitatif Komparatif',
        download_url='/download/evaluasi-metode-excel',
        headers=df.columns.tolist(),
        rows=df.fillna('').values.tolist(),
        total_rows=len(df)
    )

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 DASHBOARD 3 METODE ANALISIS SENTIMEN AKTIF!")
    print("Akses website di: http://127.0.0.1:5000")
    print("Tekan Ctrl+C untuk menghentikan server.")
    print("="*60 + "\n")
    app.run(host='127.0.0.1', port=5000, debug=False)
