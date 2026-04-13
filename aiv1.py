from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
import pandas as pd
import os
import time

app = Flask(__name__)
CORS(app)

# 1. API Ayarları
MY_API_KEY = "AIzaSyCuc7q4UA5ZPj-pCyijXEDVhEOu0RjqARY"
client = genai.Client(api_key=MY_API_KEY)

# 2. Veritabanı Yolu (Bulut ve Yerel ile uyumlu hale getirildi)
# Dosya ai.py ile aynı klasörde olmalı
DB_NAME = 'apv1.csv'

def veritabanini_yukle():
    if not os.path.exists(DB_NAME):
        print(f"HATA: {DB_NAME} bulunamadı!")
        return pd.DataFrame()
    try:
        # 900k veri için on_bad_lines='skip' kritik, bozuk satırları atlar
        data = pd.read_csv(DB_NAME, sep=',', on_bad_lines='skip')
        print(f"--- Veritabanı Aktif: {len(data)} tarif yüklendi ---")
        return data
    except Exception as e:
        print(f"Okuma hatası: {e}")
        return pd.DataFrame()

df = veritabanini_yukle()

# 3. MASTER SİSTEM TALİMATI (Detaylar korundu)
LEZZET_ASISTANI_TALIMATI = """
Rol: Sen, 'Lezzet Asistanı' uygulamasının merkezi yapay zeka beynisin. Disiplinler arası bir yetenekle; hem profesyonel bir şef, hem bir beslenme uzmanı, hem de bir veri analisti gibi davranırsın.
Görevlerin: 1. Veri Sadakati (CSV dışına çıkma), 2. Besin Analizi (NaN ise tahmin et), 3. Katı Filtreleme (Vegan/Vejetaryen/Glutensiz), 4. Matematiksel Ölçekleme (Kişi sayısına göre).
İletişim: Markdown kullan, sonuna Şefin İpucu, Besin Değeri Analizi ve Sağlık Uyarısı ekle.
"""

def ilgili_tarifleri_bul(soru):
    if df.empty: return "Veri bulunamadı."
    keywords = soru.lower().split()
    # Arama mantığını geliştiriyoruz
    mask = df['Baslik'].str.contains('|'.join(keywords), case=False, na=False) | \
           df['Malzemeler'].str.contains('|'.join(keywords), case=False, na=False)
    return df[mask].head(5).to_csv(index=False)

@app.route('/sor', methods=['POST'])
def ask_chef():
    data = request.json
    user_soru = data.get('soru', '')
    if not user_soru: return jsonify({"hata": "Soru yok"}), 400

    veriler = ilgili_tarifleri_bul(user_soru)
    modeller = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest"]
    
    for m in modeller:
        try:
            response = client.models.generate_content(
                model=m,
                config={"system_instruction": LEZZET_ASISTANI_TALIMATI, "temperature": 0.7},
                contents=f"VERİTABANI:\n{veriler}\n\nSORU: {user_soru}"
            )
            return jsonify({"cevap": response.text, "durum": "basarili"})
        except:
            continue
    return jsonify({"hata": "Sunucu yoğun"}), 503

if __name__ == '__main__':
    # Bulut sunucuları portu 'PORT' ortam değişkeninden okur. 
    # Eğer yoksa varsayılan olarak 5000'de çalışır.
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)