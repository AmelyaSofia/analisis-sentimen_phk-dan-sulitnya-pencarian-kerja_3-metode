import os
import re
import sys
import html
import pandas as pd

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(DATA_PROCESSED, exist_ok=True)

# ==============================================================================
# 1. KAMUS NORMALISASI SLANG / SINGKATAN (BAHASA INDONESIA)
# ==============================================================================
SLANG_DICT = {
    'yg': 'yang', 'dgn': 'dengan', 'krn': 'karena', 'karna': 'karena', 'utk': 'untuk',
    'sdh': 'sudah', 'udh': 'sudah', 'udah': 'sudah', 'dah': 'sudah', 'blm': 'belum',
    'blom': 'belum', 'ga': 'tidak', 'gak': 'tidak', 'nggak': 'tidak', 'ngga': 'tidak',
    'g': 'tidak', 'tdk': 'tidak', 'enggak': 'tidak', 'nda': 'tidak', 'ndak': 'tidak',
    'kagak': 'tidak', 'tak': 'tidak', 'bkn': 'bukan', 'bknnya': 'bukannya',
    'tp': 'tetapi', 'tpi': 'tetapi', 'bgt': 'sangat', 'banget': 'sangat',
    'bgtu': 'begitu', 'gini': 'begini', 'gitu': 'begitu', 'sy': 'saya', 'gw': 'saya',
    'gue': 'saya', 'ak': 'saya', 'aku': 'saya', 'lu': 'kamu', 'lo': 'kamu',
    'km': 'kamu', 'kmu': 'kamu', 'org': 'orang', 'orng': 'orang', 'bnyk': 'banyak',
    'byk': 'banyak', 'dr': 'dari', 'dpt': 'dapat', 'dapet': 'dapat', 'bs': 'bisa',
    'bsa': 'bisa', 'lg': 'lagi', 'lgi': 'lagi', 'kl': 'kalau', 'klo': 'kalau',
    'kalo': 'kalau', 'jgn': 'jangan', 'jg': 'juga', 'jga': 'juga', 'jd': 'jadi',
    'jdi': 'jadi', 'gmna': 'bagaimana', 'gmn': 'bagaimana', 'bgmn': 'bagaimana',
    'knp': 'kenapa', 'knpa': 'kenapa', 'trs': 'terus', 'trus': 'terus',
    'krja': 'kerja', 'krjaan': 'pekerjaan', 'duit': 'uang', 'cuan': 'uang',
    'nyari': 'mencari', 'nemu': 'menemukan', 'makin': 'semakin', 'mkn': 'semakin',
    'bikin': 'membuat', 'ngasih': 'memberi', 'kasih': 'memberi', 'tau': 'tahu',
    'skrg': 'sekarang', 'skrang': 'sekarang', 'hrus': 'harus', 'hrs': 'harus',
    'cmn': 'cuma', 'cuman': 'cuma', 'cm': 'cuma', 'aja': 'saja', 'aj': 'saja',
    'doang': 'saja', 'sm': 'sama', 'sma': 'sama', 'sampe': 'sampai', 'ampe': 'sampai',
    'pas': 'saat', 'pdhl': 'padahal', 'pdhal': 'padahal', 'bener': 'benar',
    'beneran': 'benaran', 'pantes': 'pantas', 'keles': 'kali', 'emang': 'memang',
    'emg': 'memang', 'kyk': 'seperti', 'kaya': 'seperti', 'kayak': 'seperti',
    'msh': 'masih', 'msih': 'masih', 'lgsg': 'langsung', 'loker': 'lowongan kerja',
    'bnget': 'sangat', 'paraaah': 'parah', 'parahhh': 'parah', 'ancur': 'hancur'
}

CONTEXT_MAPPINGS = [
    (r'\bs1\b', 'sarjana'),
    (r'\bs2\b', 'magister'),
    (r'\bs3\b', 'doktor'),
    (r'\bd3\b', 'diploma'),
]

# ==============================================================================
# 2. DAFTAR STOPWORD DENGAN WHITELIST KATA NEGASI
# ==============================================================================
NEGATION_WORDS = {
    'tidak', 'bukan', 'jangan', 'belum', 'tanpa', 'kurang', 'tiada', 'jarang',
    'sangat', 'terlalu', 'hancur', 'parah', 'susah', 'sulit', 'kalah'
}

STOPWORDS = {
    'ada', 'adalah', 'adanya', 'adapun', 'agak', 'agar', 'akan', 'akankah', 'akhir',
    'akhirnya', 'amat', 'antara', 'antaranya', 'apa', 'apabila', 'apakah', 'apalagi',
    'artinya', 'asal', 'asalkan', 'atas', 'atau', 'ataupun', 'awal', 'awalnya',
    'bagai', 'bagaikan', 'bagaimana', 'bagaimanapun', 'bagi', 'bagian', 'bahkan',
    'bahwa', 'bahwasanya', 'balik', 'bapak', 'baru', 'bawah', 'beberapa', 'begini',
    'begitu', 'belakang', 'belakangan', 'berada', 'berapa', 'berapalah', 'berbagai',
    'berikut', 'berikutnya', 'bersama', 'biasa', 'biasanya', 'bila', 'bilakah',
    'bisa', 'bisakah', 'boleh', 'buat', 'dari', 'daripada', 'demi', 'demikian',
    'dengan', 'depan', 'di', 'dia', 'dialah', 'diantara', 'diantaranya', 'diri',
    'dirinya', 'disini', 'dong', 'dulu', 'hal', 'hampir', 'hanya', 'hari', 'harus',
    'hendak', 'hingga', 'ia', 'ialah', 'ibu', 'ikut', 'ini', 'inikah', 'inilah',
    'itu', 'itukah', 'itulah', 'jadi', 'jadilah', 'jauh', 'jika', 'jikalau', 'juga',
    'jumlah', 'justru', 'kala', 'kalau', 'kalian', 'kami', 'kamu', 'kan', 'kapan',
    'karena', 'kasus', 'kata', 'katanya', 'ke', 'keadaan', 'kedua', 'keduanya',
    'keluar', 'kembali', 'kemudian', 'kenapa', 'kepada', 'ketika', 'kini', 'kira',
    'kita', 'kok', 'lagi', 'lagian', 'lah', 'lain', 'lainnya', 'lalu', 'lama',
    'lanjut', 'lewat', 'maka', 'makanya', 'malah', 'mana', 'masih', 'masing',
    'mau', 'maupun', 'melalui', 'melihat', 'memang', 'memberi', 'membuat', 'menjadi',
    'menurut', 'mereka', 'nah', 'namun', 'oleh', 'pada', 'padahal', 'para', 'pula',
    'pun', 'saat', 'saja', 'saling', 'sama', 'sampai', 'sana', 'satu', 'saya',
    'sebab', 'sebagai', 'sebagaimana', 'sebelum', 'sedang', 'sedangkan', 'segera',
    'sehingga', 'sejak', 'sekali', 'sekarang', 'sekitar', 'selain', 'selalu',
    'selama', 'seluruh', 'sementara', 'semua', 'semula', 'seperti', 'sering',
    'serta', 'sesi', 'sesuai', 'sesuatu', 'setelah', 'setiap', 'siapa', 'sini',
    'suatu', 'sudah', 'tadi', 'tahu', 'tentang', 'tentu', 'terhadap', 'tersebut',
    'terus', 'tetapi', 'tiba', 'tuh', 'untuk', 'waktu', 'walau', 'walaupun',
    'ya', 'yaitu', 'yakni', 'yang'
}

EFFECTIVE_STOPWORDS = STOPWORDS - NEGATION_WORDS

# ==============================================================================
# 3. FUNGSI PREPROCESSING
# ==============================================================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'[@#]\w+', ' ', text)
    text = re.sub(r'\b\d{1,2}:\d{2}\b', ' ', text)
    
    emoji_pattern = re.compile(
        "["
        "\U00010000-\U0010ffff"
        "\u2600-\u26ff"
        "\u2700-\u27bf"
        "\u2b50"
        "\u200d"
        "\ufe0f"
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub(' ', text)
    
    lower_temp = text.lower()
    for pattern, replacement in CONTEXT_MAPPINGS:
        lower_temp = re.sub(pattern, replacement, lower_temp)
    text = lower_temp
    
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def case_folding(text):
    return text.lower()

def tokenize(text):
    return text.split()

def normalize_slang(tokens):
    return [SLANG_DICT.get(token, token) for token in tokens]

def remove_stopwords(tokens):
    return [token for token in tokens if (token not in EFFECTIVE_STOPWORDS) and (len(token) > 1)]

def preprocess_comment(text):
    cleaned = clean_text(text)
    folded = case_folding(cleaned)
    tokens = tokenize(folded)
    normalized_tokens = normalize_slang(tokens)
    filtered_tokens = remove_stopwords(normalized_tokens)
    final_text = " ".join(filtered_tokens)
    return final_text, filtered_tokens

def run_preprocessing(input_file="hasil_scraper.csv", output_file="hasil_preprocessing.csv"):
    input_path = os.path.join(DATA_RAW, input_file)
    output_path = os.path.join(DATA_PROCESSED, output_file)
    
    print(f"[*] Membaca data mentah dari: {input_path}")
    if not os.path.exists(input_path):
        print(f"[!] File {input_path} tidak ditemukan!")
        return None
        
    df = pd.read_csv(input_path)
    print(f"[*] Total data mentah: {len(df)} baris")
    
    results = [preprocess_comment(comment) for comment in df['comment']]
    df['clean_comment'] = [res[0] for res in results]
    df['tokens'] = [str(res[1]) for res in results]
    
    initial_len = len(df)
    df_cleaned = df[df['clean_comment'].str.strip() != ""].copy()
    dropped_count = initial_len - len(df_cleaned)
    
    df_cleaned.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*60)
    print("HASIL PREPROCESSING:")
    print(f"- Data awal        : {initial_len} baris")
    print(f"- Dihapus (kosong) : {dropped_count} baris")
    print(f"- Data bersih final: {len(df_cleaned)} baris")
    print(f"- File tersimpan di: {output_path}")
    print("="*60)
    return df_cleaned

if __name__ == "__main__":
    run_preprocessing()
