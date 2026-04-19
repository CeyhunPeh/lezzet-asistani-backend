from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
import pandas as pd
import os
import re
from sqlalchemy import create_engine
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)



API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("!!! KRİTİK HATA: GEMINI_API_KEY bulunamadı. Render Environment Variables kısmını kontrol et!")

client = genai.Client(api_key=API_KEY)



load_dotenv(override=True)
DB_URL = os.environ.get("DATABASE_URL")

if DB_URL and DB_URL.startswith("postgres://"):
    DB_URL = DB_URL.replace("postgres://", "postgresql://", 1)


engine = create_engine(DB_URL) if DB_URL else None

def veritabanini_yukle():
    if not engine:
        print("!!! HATA: DATABASE_URL bulunamadı. Render Environment Variables kısmına Neon linkini ekle!")
        return pd.DataFrame()
    
    try:
        print("🧠 Lezzet Asistanı hafızasını buluttan indiriyor...")
        # Tüm veriyi Neon bulutundan çek
        data = pd.read_sql("SELECT * FROM tarifler", engine)
        
        # Orijinal kodundaki bellek optimizasyonlarını (kategori ve string atamalarını) koruyoruz
        if 'Kategori' in data.columns:
            data['Kategori'] = data['Kategori'].astype('category')
        for col in ['Baslik', 'Malzemeler', 'Malzemelerin Miktari']:
            if col in data.columns:
                data[col] = data[col].astype('string')
                
        print(f"--- SİSTEM AKTİF: {len(data)} Tarif Başarıyla Belleğe Alındı ---")
        return data
    except Exception as e:
        print(f"!!! Veritabanı okuma hatası: {e}")
        return pd.DataFrame()

df = veritabanini_yukle()

LEZZET_ASISTANI_TALIMATI = """
--- ROL VE KİMLİK (EN KRİTİK) ---
Sen Lezzet Dünyası uygulamasının neşeli, samimi ve uzman chatbotu TOMBİK ŞEF'sin. 20.521 tariflik dev bir arşivin başındasın. Daima samimi bir selamla başla (Selam evlat, Hoş geldin mutfak dostu vb.) ve asla kimliğini bırakma.

--- VERİ YÖNETİMİ VE KURALLAR ---
1. VERİ SADAKATİ: Sadece sana gelen 'VERİTABANI VERİLERİ' içindeki tarifleri sun. Dışarıdan tarif uydurma.
2. APERATİF HATASI: Sana gelen liste sadece arama sonuçlarıdır. Eğer listede sadece aperatif varsa, bu 'arşivde sadece aperatif var' demek DEĞİLDİR. Kullanıcıyı 'Şu anki sonuçlar bunlar ama başka bir şey de arayabiliriz' diye yönlendir.
3. MATEMATİKSEL ÖLÇEKLEME: Kullanıcı kişi sayısı verirse 'Malzemelerin Miktari' sütunundaki rakamları oranla. 'Göz kararı' gibi ifadeleri elleme.
4. EKSİK VERİ: Kalori/Besin değerleri boşsa, malzemelerden tahmin yürüt ve '(Tahmini değerdir)' yaz.
5. FİLTRELEME: Vegan, Vejetaryen ve Glutensiz kurallarına %100 uy.

--- GÖRSEL VE YAZIM STANDARTLARI ---
- MARKDOWN YASAK: Yıldız (*), çift yıldız (**) veya kare (#) kesinlikle kullanma. Metni kalın yapma.
- BAŞLIKLAR: Sadece '--- BAŞLIK ADI ---' formatını kullan.
- LİSTELEME: Sadece kısa tire (-) kullan.
- LİNK YASAK: Tarif linklerini asla paylaşma.

--- YANIT YAPISI (Sadece Tarif Varsa) ---
Tarif bitince şu 3 başlığı mutlaka ekle:
--- ŞEFİN İPUCU ---
--- BESİN DEĞERİ ANALİZİ ---
--- SAĞLIK UYARISI ---

--- KAPANIŞ SORUSU ---
Her cevabı bağlama uygun, doğal bir soruyla bitir. (Örn: Yanına ne pişirelim? Malzeme değiştirmek ister misin?)
"""

def ilgili_tarifleri_bul(soru):
    if df.empty: return "VERİTABANI_ERİŞİM_HATASI"
    
    soru_temiz = soru.lower().strip()
    # Selamlaşma kontrolü
    selamlar = ['selam', 'merhaba', 'meraba', 'naber', 'hi', 'hello', 'tombik', 'şef']
    if any(s == soru_temiz for s in selamlar) or len(soru_temiz) < 3:
        return "GREETING_MODE: Kullanıcı selam verdi, ona Tombik Şef olarak sıcak bir karşılama yap."

    keywords = soru_temiz.split()
    
    # 1. KATEGORİ ÖNCELİĞİ (Et yemeği vb. aramalar için)
    # Kategori sütununda birebir veya kısmi eşleşme ara
    kategori_mask = df['Kategori'].str.contains(soru_temiz, case=False, na=False)
    kategori_sonuclari = df[kategori_mask]

    # 2. BAŞLIK VE MALZEME ARAMASI
    genel_mask = df['Baslik'].str.contains('|'.join(keywords), case=False, na=False) | \
                 df['Malzemeler'].str.contains('|'.join(keywords), case=False, na=False)
    genel_sonuclar = df[genel_mask]

    # 3. HİYERARŞİK BİRLEŞTİRME
    # Önce kategori sonuçlarını, sonra genel sonuçları koyuyoruz
    final_df = pd.concat([kategori_sonuclari, genel_sonuclar]).drop_duplicates().head(20)
    
    if final_df.empty:
        return "UYARI: Veritabanında tam sonuç yok, kullanıcının talebine yakın genel öneriler yap."
        
    return final_df.to_csv(index=False)


def markdown_temizle(text):
    
    return text.replace("**", "").replace("*", "- ").replace("#", "")


@app.route('/sor', methods=['POST'])
def ask_chef():
    data = request.json
    user_soru = data.get('soru', '')
    
    if not user_soru:
        return jsonify({"hata": "Lütfen bir soru sorunuz.", "durum": "hata"}), 400

    print(f"Gelen İstek: {user_soru}")
    veriler = ilgili_tarifleri_bul(user_soru)
    

    modeller = [
        
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite-preview",
        "gemma-4-31b-it",
        "gemma-3-27b-it",
        "gemini-2.5-pro",
        "gemini-pro-latest",
        "gemini-1.5-flash-latest",  
        "gemini-1.5-flash"
    ]
    
    for m in modeller:
        try:
            print(f"--> {m} modeli ile iletişim kuruluyor...")
            response = client.models.generate_content(
                model=m,
                config={
                    "system_instruction": LEZZET_ASISTANI_TALIMATI,
                    "temperature": 0.6 
                },
                contents=f"VERİTABANI VERİLERİ (CSV):\n{veriler}\n\nKULLANICI SORUSU: {user_soru}"
            )
            
            
            temiz_cevap = markdown_temizle(response.text)
            
            print(f"✓ {m} başarılı bir cevap üretti.")
            return jsonify({
                "cevap": temiz_cevap,
                "durum": "basarili",
                "model": m
            })
            
        except Exception as e:
            print(f"!!! {m} hatası: {str(e)}")
            
            if m == modeller[-1]:
                return jsonify({
                    "hata": "Sunucu yoğun", 
                    "debug_notu": str(e),
                    "yardim": "Yeni bir API Key aldığınızdan ve Render Environment Variables kısmına eklediğinizden emin olun."
                }), 503
            
            continue
        
@app.route('/', methods=['GET'])
def uyanik_kal():
    
    return "Lezzet Asistanı 7/24 Uyanık ve Görev Başında!", 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)