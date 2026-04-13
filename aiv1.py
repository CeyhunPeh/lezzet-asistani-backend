from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
import pandas as pd
import os
import time

app = Flask(__name__)
CORS(app)

# 1. API Ayarları - Güvenli Yöntem
# Render panelindeki 'Environment Variables' kısmından okur.
API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyCuc7q4UA5ZPj-pCyijXEDVhEOu0RjqARY")
client = genai.Client(api_key=API_KEY)

# 2. Veritabanı Yolu
DB_NAME = 'apv1.csv'

def veritabanini_yukle():
    if not os.path.exists(DB_NAME):
        print(f"!!! HATA: {DB_NAME} bulunamadı! Dosya ismini ve GitHub'daki varlığını kontrol et.")
        return pd.DataFrame()
    try:
        data = pd.read_csv(DB_NAME, sep=',', on_bad_lines='skip')
        print(f"--- Veritabanı Aktif: {len(data)} tarif başarıyla hafızaya alındı ---")
        return data
    except Exception as e:
        print(f"!!! Veritabanı okuma hatası: {e}")
        return pd.DataFrame()

df = veritabanini_yukle()

# 3. MASTER SİSTEM TALİMATI (En Detaylı Hali)
LEZZET_ASISTANI_TALIMATI = """
Rol: Sen, 'Lezzet Asistanı' uygulamasının merkezi yapay zeka beynisin. Disiplinler arası bir yetenekle; hem profesyonel bir şef, hem bir beslenme uzmanı, hem de bir veri analisti gibi davranırsın.

Görevlerin ve Protokoller:
1. Veri Sadakati: Sadece sana sağlanan veritabanındaki tarifleri sunmalısın. 
2. Besin Analizi: Eğer veritabanında değerler 'NaN' ise malzemelere bakarak yaklaşık tahmin yap ve '(Tahmini değerdir)' notunu ekle.
3. Filtreleme: Vegan, Vejetaryen ve Glutensiz filtrelerine %100 sadık kal.
4. Operasyonel Analiz: Hazırlanış metnine göre zorluk seviyesini (Kolay/Orta/Zor) belirle.
5. Matematiksel Ölçekleme: Malzeme miktarlarını kullanıcı kişi sayısına göre hesapla.

İletişim Standartları:
- Markdown formatında, başlıklar ve listelerle cevap ver.
- Sonunda mutlaka: 1. Şefin İpucu, 2. Besin Değeri Analizi, 3. Sağlık Uyarısı ekle.
- 'Link' sütunundaki adresi 'Daha Fazla Detay İçin Tıkla' olarak paylaş.
"""

def ilgili_tarifleri_bul(soru):
    if df.empty: return "Veritabanı şu an boş veya yüklenemedi."
    keywords = soru.lower().split()
    # Arama motoru mantığı: Başlık veya malzemelerde kelimeleri ara
    mask = df['Baslik'].str.contains('|'.join(keywords), case=False, na=False) | \
           df['Malzemeler'].str.contains('|'.join(keywords), case=False, na=False)
    
    sonuc = df[mask].head(7).to_csv(index=False)
    if not sonuc: return "İlgili tarif bulunamadı, genel mutfak bilgini kullanabilirsin."
    return sonuc

@app.route('/sor', methods=['POST'])
def ask_chef():
    data = request.json
    user_soru = data.get('soru', '')
    
    if not user_soru:
        return jsonify({"hata": "Lütfen bir soru sorunuz.", "durum": "hata"}), 400

    print(f"Yeni istek geldi: {user_soru}")
    veriler = ilgili_tarifleri_bul(user_soru)
    
    # Sırasıyla modelleri deniyoruz
    modeller = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest"]
    
    for m in modeller:
        try:
            print(f"--> {m} deneniyor...")
            response = client.models.generate_content(
                model=m,
                config={
                    "system_instruction": LEZZET_ASISTANI_TALIMATI,
                    "temperature": 0.7
                },
                contents=f"VERİTABANI VERİLERİ:\n{veriler}\n\nKULLANICI SORUSU: {user_soru}"
            )
            print(f"✓ {m} başarılı cevap üretti.")
            return jsonify({
                "cevap": response.text,
                "durum": "basarili",
                "model": m
            })
        except Exception as e:
            # Burası çok önemli: Hatanın nedenini Render loglarına yazdırır
            print(f"!!! {m} hatası: {str(e)}")
            if m == modeller[-1]:
                # Tüm modeller bittiyse gerçek hatayı JSON olarak döndür
                return jsonify({
                    "hata": "Sunucu yoğun", 
                    "debug_notu": str(e),
                    "mesaj": "Google API bağlantı hatası. Lütfen API key kontrolü yapın."
                }), 503
            continue

# Render için Port ayarı
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)