from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
import pandas as pd
import os
import time

app = Flask(__name__)
CORS(app)

# 1. API Ayarları - Tam Güvenlik Protokolü
# Kodun içine asla manuel anahtar yazmıyoruz. 
# Render panelindeki 'Environment' sekmesine eklediğin anahtarı çeker.
API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    print("!!! KRİTİK HATA: GEMINI_API_KEY bulunamadı. Render Environment Variables kısmını kontrol et!")

client = genai.Client(api_key=API_KEY)

# 2. Veritabanı Yönetimi
DB_NAME = 'aperatif_veritabani.csv'

def veritabanini_yukle():
    if not os.path.exists(DB_NAME):
        print(f"!!! HATA: {DB_NAME} dosyası klasörde yok. GitHub yüklemesini kontrol et.")
        return pd.DataFrame()
    try:
        data = pd.read_csv(DB_NAME, sep=',', on_bad_lines='skip')
        print(f"--- Veritabanı Aktif: {len(data)} tarif başarıyla yüklendi ---")
        return data
    except Exception as e:
        print(f"!!! Veritabanı okuma hatası: {e}")
        return pd.DataFrame()

df = veritabanini_yukle()

# 3. MASTER SİSTEM TALİMATI (Şef & Analist Protokolü)
LEZZET_ASISTANI_TALIMATI = """
Rol: Sen, 'Lezzet Asistanı' uygulamasının merkezi yapay zeka beynisin. Disiplinler arası bir yetenekle; hem profesyonel bir şef, hem bir beslenme uzmanı, hem de bir veri analisti gibi davranırsın.

Görevlerin ve Operasyonel Protokoller:
1. Veri Sadakati: Sadece sana sağlanan veritabanındaki gerçek tarifleri sunmalısın. 
2. Besin Analizi: Veritabanında değerler 'NaN' (boş) ise malzemelere bakarak yaklaşık tahmin yap ve yanına '(Tahmini değerdir)' notunu ekle.
3. Kesin Filtreleme: Vegan, Vejetaryen ve Glutensiz filtrelerine %100 sadık kal. Şüpheli malzemelerde kullanıcıyı uyar.
4. Operasyonel Analiz: Hazırlanış metnine göre zorluk seviyesini (Kolay/Orta/Zor) belirle.
5. Matematiksel Ölçekleme: Malzeme miktarlarını kullanıcının belirttiği kişi sayısına göre matematiksel olarak hesapla.
6. Yorumlayıcı Destek: Tarifin linkini asla paylaşma. Kullanıcı tarif hakkında soru sorarsa, veritabanındaki malzemeleri ve hazırlanış adımlarını bir şef hassasiyetiyle yorumlayarak yardımcı ol. (Örn: Malzeme değişimi, pişirme teknikleri vb.)

İletişim ve Görsel Standartlar (Kritik):
- GÖRSEL SADELİK: Gereksiz yıldız (*) ve (**) kullanımından kaçın.
- Okunabilirliği madde işaretleri yerine net satır boşlukları ve sadece '---' başlıkları kullanarak sağla.
- Yanıt sonunda mutlaka şu 3 bölümü sade bir şekilde ekle: 
  ### Şefin İpucu
  ### Besin Değeri Analizi
  ### Sağlık Uyarısı
- Kullanıcıyı sohbeti devam ettirmeye teşvik et (Örn: 'Bu tarifteki bir malzemeyi değiştirmek ister misin?').
"""
def ilgili_tarifleri_bul(soru):
    if df.empty: return "Veritabanı erişilemez durumda."
    keywords = soru.lower().split()
    # Arama: Başlık veya Malzemeler sütununda anahtar kelimeleri tara
    mask = df['Baslik'].str.contains('|'.join(keywords), case=False, na=False) | \
           df['Malzemeler'].str.contains('|'.join(keywords), case=False, na=False)
    
    sonuc = df[mask].head(7).to_csv(index=False)
    return sonuc if sonuc else "İlgili tarif bulunamadı."

@app.route('/sor', methods=['POST'])
def ask_chef():
    data = request.json
    user_soru = data.get('soru', '')
    
    if not user_soru:
        return jsonify({"hata": "Lütfen bir soru sorunuz.", "durum": "hata"}), 400

    print(f"Gelen İstek: {user_soru}")
    veriler = ilgili_tarifleri_bul(user_soru)
    
    # Model Fallback (Hata durumunda sırayla deneme)
    modeller = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest"]
    
    for m in modeller:
        try:
            print(f"--> {m} modeli ile iletişim kuruluyor...")
            response = client.models.generate_content(
                model=m,
                config={
                    "system_instruction": LEZZET_ASISTANI_TALIMATI,
                    "temperature": 0.7
                },
                contents=f"VERİTABANI VERİLERİ (CSV):\n{veriler}\n\nKULLANICI SORUSU: {user_soru}"
            )
            print(f"✓ {m} başarılı bir cevap üretti.")
            return jsonify({
                "cevap": response.text,
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