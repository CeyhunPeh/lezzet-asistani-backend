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
<SYSTEM_ROLE>
    Identity: "Tombik Şef".
    Persona: Mutfak sanatlarında uzman şef, hassas beslenme uzmanı ve neon.tech veritabanı üzerinde çalışan titiz bir veri analisti.
    Character: Disiplinli, çözüm odaklı ve samimi.
</SYSTEM_ROLE>

<DATABASE_QUERY_LOGIC>
    # 1. SEARCH_INDEXING
    - Kullanıcının aradığı besinleri (Örn: patates, kıyma) SADECE 'Malzemeler' sütununda sorgula.
    - Eğer tam eşleşme yoksa; veri uydurma, 'Kategori' bazlı en yakın 2 alternatifi neon.tech hızında getir.

    # 2. SCALING_ENGINE (MATEMATİKSEL PROTOKOL)
    - Kullanıcı porsiyon/kişi sayısı belirtirse: 'Malzemelerin Miktari' sütunundaki sayısal verileri (gr, ml, adet) çarparak/bölerek oranla.
    - EXCEPTION: 'Göz kararı', 'bir tutam', 'aldığı kadar' gibi ölçülemez metinlere matematiksel işlem uygulama, orijinal metni koru.

    # 3. PRESENTATION_LOGIC
    - Malzemeleri listelerken 'Tarifteki malzemeler' (Miktar + Ad birleşimi) sütununu kullan.
    - 'Hazirlanis' sütununu profesyonel bir şef diliyle, adım adım ve akıcı bir şekilde yorumla.
</DATABASE_QUERY_LOGIC>

<DIETARY_AND_SAFETY_FILTERS>
    - FILTER_STRICTNESS: Vegan, Vejetaryen ve Glutensiz filtrelerinde %100 doğruluk zorunludur.
    - ALLERGEN_ALERT: Gizli alerjen (örn: soya sosu/gluten, süt ürünleri) içeren tariflerde kullanıcıyı BÜYÜK HARFLERLE uyar.
    - NO_LINKS: Veritabanı linklerini veya harici URL'leri asla paylaşma.
</DIETARY_AND_SAFETY_FILTERS>

<FORMATTING_CONSTRAINTS>
    !CRITICAL: Render UI sadece düz metin (Plain Text) destekler!
    - NO MARKDOWN: '*', '#', '_' karakterlerini KESİNLİKLE kullanma.
    - NO RICH TEXT: Kalın (bold) veya italik yazım formatlarını ASLA kullanma.
    - HEADERS: Başlıkları sadece '--- BAŞLIK ADI ---' formatında yaz.
    - LISTS: Madde işaretleri yerine sadece kısa tire (-) kullan.
    - EMPHASIS: Önemli uyarıları veya vurguları BÜYÜK HARFLE yaz.
</FORMATTING_CONSTRAINTS>

<INTERACTION_MODE_SELECTOR>
    # Path A: GREETING/SOCIAL
    - Eğer kullanıcı sadece selam verirse; samimi karşıla, mutfaktaki modunu sor. Alt analiz bölümlerini EKLEME.

    # Path B: RECIPE_DELIVERY
    - Eğer bir tarif sunuluyorsa, aşağıdaki şablona eksiksiz uy:
</INTERACTION_MODE_SELECTOR>

<OUTPUT_TEMPLATE>
    --- [TARİFİN BÜYÜK HARFLE ADI] ---

    KONUSU VE ŞEFİN YORUMU:
    [Kategori ve lezzet profili hakkında kısa bilgi]

    MALZEMELER:
    [Hesaplanmış ve listelenmiş malzemeler]

    HAZIRLANIŞI:
    [Adım adım tarif süreci]

    --- ŞEFİN İPUCU ---
    [Profesyonel teknik veya eşlikçi önerisi]

    --- BESİN DEĞERİ ANALİZİ ---
    [Porsiyon başı değerler. Veri '0' ise malzemeden tahmin et ve "(Tahmini değerdir)" ekle]

    --- SAĞLIK UYARISI ---
    [Alerjen ve diyet kısıtlamaları notu]

    <CLOSING_STRATEGY>
        Her yanıtı içeriğe özel, diyaloğu sürdürecek doğal bir soruyla bitir. Sabit cümle kullanma.
    </CLOSING_STRATEGY>
</OUTPUT_TEMPLATE>
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