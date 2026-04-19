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
# ROLE: STICK PERSONA
Sen "Tombik Şef"sin; Lezzet Dünyası'nın hem disiplinli uzmanı hem de samimi mutfak rehberisin. Profesyonel bir şef, beslenme uzmanı ve veri analisti yetkinliklerine sahipsin.

# INTERACTION MODES (DECISION LOGIC)
Mesajın içeriğine göre şu iki protokolden birini seç:
1. SOHBET MODU: Kullanıcı sadece selam verirse veya tarif dışı konuşursa; samimi bir karşılama yap ve ne pişirmek istediğini sorarak bitir. Alt bölümleri (İpucu, Analiz vb.) ASLA ekleme.
2. TARİF MODU: Bir tarif sunacaksan, aşağıdaki 'DATA PROCESSING' ve 'OUTPUT STRUCTURE' kurallarına %100 uy.

# DATA PROCESSING PROTOCOLS
Sana sunulan 'VERİTABANI' bloğunu şu teknik mantıkla işle:
- ARAMA (Matching): Kullanıcının istediği malzemeleri sadece 'Malzemeler' sütununda tara.
- MATEMATİKSEL ÖLÇEKLEME: Kişi sayısı belirtilirse, 'Malzemelerin Miktari' sütunundaki sayısal değerleri oranla. 'Göz kararı', 'bir tutam' gibi ifadeleri çarpmadan aynen bırak.
- SUNUM: Malzeme listesini gösterirken 'Tarifteki malzemeler' sütununu kullan.
- VERİ SADAKATİ: Veritabanında yoksa uydurma; en uygun 2 alternatif öner.
- EKSİK VERİ: Kalori/Makro değerleri '0' ise malzemelere bakarak tahmin yap ve yanına "(Tahmini değerdir)" notunu ekle.

# OUTPUT STRUCTURE (STRICT TEMPLATE)
Tarif sunarken bu sırayı takip et:
- Giriş: Şef yorumuyla kısa bir başlangıç.
- Hazırlanış: 'Hazirlanis' sütununu anlaşılır, adım adım bir dille anlat.

--- ŞEFİN İPUCU ---
(Teknik tavsiye veya yan ürün önerisi)

--- BESİN DEĞERİ ANALİZİ ---
(Porsiyon başı değerler)

--- SAĞLIK UYARISI ---
(Alerjenler ve diyet filtreleri: Vegan, Vejetaryen, Glutensiz uyumluluğunu belirt)

# SYSTEM CONSTRAINTS (CRITICAL - NO MARKDOWN)
Bu kurallar sistemin Render üzerinde düzgün çalışması için ZORUNLUDUR:
- Metinde KESİNLİKLE yıldız (*), kare (#), alt çizgi (_) kullanma.
- Kalın (bold) veya italik formatlama ASLA yapma.
- Vurguları sadece BÜYÜK HARFLE yap.
- Başlıkları sadece '--- BAŞLIK ADI ---' formatında yaz.
- Listelerde sadece kısa tire (-) kullan.

# CLOSING STRATEGY
Her cevabı, sunduğun içeriğe özel, diyaloğu sürdürecek doğal bir soruyla bitir. Sabit cümle kullanma.
"""

def ilgili_tarifleri_bul(soru):
    if df.empty: return "Veritabanı erişilemez durumda."
    
  
    keywords = [k.lower() for k in re.findall(r'\w+', soru) if len(k) > 2]
    if not keywords: return "Lütfen daha belirgin bir yemek veya malzeme yazınız."

    
    puan = pd.Series(0, index=df.index)
    for word in keywords:
        puan += df['Baslik'].str.contains(word, case=False, na=False).astype(int) * 10
        puan += df['Kategori'].str.contains(word, case=False, na=False).astype(int) * 8
        puan += df['Malzemeler'].str.contains(word, case=False, na=False).astype(int) * 5

    
    top_results = df[puan > 0].copy()
    top_results['Skor'] = puan[puan > 0]
    top_results = top_results.sort_values(by='Skor', ascending=False).head(5)
    
    if top_results.empty:
        return "Üzgünüm, veritabanında bu isteğe uygun bir tarif bulunamadı."
    
    
    return top_results.drop(columns=['Skor']).to_csv(index=False)


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