import os
import re
import urllib.parse
import pandas as pd
from googleapiclient.discovery import build
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(DATA_RAW, exist_ok=True)

# Load environment variables
load_dotenv()
API_KEY = "AIzaSyA5-WnTjBW9kzNqOdfqOOQ3CFhR9zQogsg"

if not API_KEY or API_KEY.endswith('...'):
    print("⚠️ PERINGATAN: YouTube API Key sepertinya belum lengkap di file .env.")

def extract_video_id(url_or_id):
    """
    Ekstrak video ID dari berbagai format URL YouTube atau string ID langsung.
    """
    url_or_id = url_or_id.strip()
    if re.match(r'^[a-zA-Z0-9_-]{11}$', url_or_id):
        return url_or_id
    
    parsed = urllib.parse.urlparse(url_or_id)
    if parsed.hostname in ('www.youtube.com', 'youtube.com', 'm.youtube.com'):
        if parsed.path == '/watch':
            queries = urllib.parse.parse_qs(parsed.query)
            return queries.get('v', [None])[0]
        elif parsed.path.startswith('/shorts/') or parsed.path.startswith('/embed/'):
            return parsed.path.split('/')[2]
    elif parsed.hostname == 'youtu.be':
        return parsed.path[1:].split('?')[0]
        
    return url_or_id

def get_video_comments(youtube, video_id, max_results=100):
    """
    Mengambil komentar dari satu video YouTube hingga batas max_results.
    """
    print(f"\n[+] Mengambil komentar untuk video ID: {video_id} (Target: {max_results} komentar)...")
    comments = []
    
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(max_results, 100),
            textFormat="plainText",
            order="relevance"
        )
        response = request.execute()
        
        while response and len(comments) < max_results:
            for item in response.get('items', []):
                snippet = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'video_id': video_id,
                    'author': snippet.get('authorDisplayName', ''),
                    'comment': snippet.get('textDisplay', ''),
                    'like_count': snippet.get('likeCount', 0),
                    'published_at': snippet.get('publishedAt', '')
                })
                
                if len(comments) >= max_results:
                    break
            
            if 'nextPageToken' in response and len(comments) < max_results:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    pageToken=response['nextPageToken'],
                    maxResults=min(max_results - len(comments), 100),
                    textFormat="plainText",
                    order="relevance"
                )
                response = request.execute()
            else:
                break
                
        print(f"    -> Berhasil mengambil {len(comments)} komentar dari video {video_id}.")
        
    except Exception as e:
        print(f"    -> Terjadi error pada video {video_id}: {e}")
        
    return comments

def scrape_multiple_videos(video_sources, max_per_video=100, output_filename="hasil_scraper.csv"):
    """
    Scraping komentar dari beberapa link/ID video dan menyimpannya ke folder data/raw/.
    """
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    all_comments = []
    
    for idx, source in enumerate(video_sources, start=1):
        v_id = extract_video_id(source)
        if not v_id:
            print(f"[-] Format URL/ID tidak valid: {source}")
            continue
            
        print(f"\n--- Memproses Video ke-{idx}: {v_id} ---")
        video_comments = get_video_comments(youtube, v_id, max_results=max_per_video)
        all_comments.extend(video_comments)
        
    df_all = pd.DataFrame(all_comments)
    
    if not df_all.empty:
        output_path = os.path.join(DATA_RAW, output_filename)
        df_all.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print("\n" + "="*50)
        print(f"SELESAI! Total data berhasil dikumpulkan: {len(df_all)} baris.")
        print(f"Data disimpan ke: {output_path}")
        print("Distribusi komentar per video:")
        print(df_all['video_id'].value_counts())
        print("="*50)
    else:
        print("\n[!] Tidak ada komentar yang berhasil diambil.")
        
    return df_all

if __name__ == "__main__":
    TARGET_VIDEOS = [
        "https://www.youtube.com/watch?v=Zvl7wH2eIC0&t=2s",
        "https://www.youtube.com/watch?v=3_6QjUNo9eY",
        "https://www.youtube.com/watch?v=4IBxusLCXsw"
    ]
    COMMENTS_PER_VIDEO = 100
    scrape_multiple_videos(TARGET_VIDEOS, max_per_video=COMMENTS_PER_VIDEO)
