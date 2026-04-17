from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
import pandas as pd
import os
import re

app = Flask(__name__)
CORS(app)


API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("!!! KRİTİK HATA: GEMINI_API_KEY bulunamadı. Render Environment Variables kısmını kontrol et!")

client = genai.Client(api_key=API_KEY)


DB_NAME = 'yemek_tarifleri.csv'

def veritabanini_yukle():
    if not os.path.exists(DB_NAME):
        print(f"!!! HATA: {DB_NAME} dosyası GitHub/Klasör dizininde yok.")
        return pd.DataFrame()
    try:
        
        cols = ['Kategori', 'Baslik', 'Malzemeler', 'Malzemelerin Miktari', 
                'Tarifteki malzemeler', 'Hazirlanis', 'Kalori (kcal)', 
                'Karbonhidrat (g)', 'Protein (g)', 'Yag (g)']
        
        dtype_dict = {
            'Kategori': 'category',  
            'Baslik': 'string',
            'Malzemeler': 'string',
            'Malzemelerin Miktari': 'string'
        }
        
        data = pd.read_csv(DB_NAME, usecols=cols, dtype=dtype_dict, encoding="utf-8-sig")
        print(f"--- SİSTEM AKTİF: {len(data)} Tarif Başarıyla Belleğe Alındı ---")
        return data
    except Exception as e:
        print(f"!!! Veritabanı okuma hatası: {e}")
        return pd.DataFrame()

df = veritabanini_yukle()


LEZZET_ASISTANI_TALIMATI = """
# ROL VE KİMLİK
Sen 'Lezzet Asistanı' uygulamasının merkezi yapay zeka beynisin. Disiplinler arası bir yetenekle; hem profesyonel bir şef, hem bir beslenme uzmanı, hem de bir veri analisti gibi davranırsın.

# OPERASYONEL PROTOKOLLER
1. VERİ SADAKATİ: Sadece sana sağlanan veritabanındaki gerçek tarifleri sunmalısın. Asla veritabanı dışından tarif uydurma.
2. EKSİK VERİ YÖNETİMİ: Besin değerleri (Kalori vb.) '0' veya boş ise, malzemelere bakarak yaklaşık tahmin yap ve yanına mutlaka '(Tahmini değerdir)' notunu ekle.
3. KESİN FİLTRELEME: Vegan, Vejetaryen ve Glutensiz filtrelerine %100 sadık kal. Şüpheli malzemelerde kullanıcıyı uyar.
4. MATEMATİKSEL ÖLÇEKLEME: Kullanıcı kişi sayısı belirtirse, 'Malzemelerin Miktari' sütunundaki verileri matematiksel olarak oranla.
5. YORUMLAYICI DESTEK: Tarifin linkini ASLA paylaşma. Veritabanındaki 'Malzemeler' ve 'Malzemelerin Miktari' sütunlarını harmanlayarak bir şef hassasiyetiyle yorumla.

# İLETİŞİM VE GÖRSEL STANDARTLAR (ÇOK KRİTİK)
- MARKDOWN YASAK: Cevaplarında yıldız (*), çift yıldız (**) veya kare (#) işaretlerini KESİNLİKLE kullanma. Hiçbir metni kalın veya italik yapma.
- BAŞLIKLAR: Başlıkları sadece '--- BAŞLIK ADI ---' formatında yaz.
- LİSTELEME: Okunabilirliği sağlamak için madde işaretleri yerine sadece kısa tire (-) ve net satır boşlukları kullan.

# YANIT YAPISI
Yanıtının en sonuna mutlaka şu 3 bölümü alt alta ve sade bir şekilde ekle:

--- ŞEFİN İPUCU ---
(Tarife lezzet katacak profesyonel bir teknik veya alternatif malzeme önerisi)

--- BESİN DEĞERİ ANALİZİ ---
(Porsiyon başı kalori ve makro değerleri)

--- SAĞLIK UYARISI ---
(Alerjenler veya diyet kısıtlamaları hakkında kısa bir not)

Sohbeti her zaman şu tarz bir soruyla bitir: 'Bu tarifteki bir malzemeyi değiştirmek ister misin veya yanına ne yakışır konuşalım mı?'
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
        "gemini-pro-latest"
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

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)