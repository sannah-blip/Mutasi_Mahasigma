import pypdf
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import io
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

st.set_page_config(page_title="Smart Financial Assistant", layout="wide")
st.title("Smart Financial Assistant")

# ==========================================
# 1. FUNGSI EKSTRAKSI PDF & PDF LAPORAN
# ==========================================
def ekstrak_dari_pdf(file_pdf):
    reader = pypdf.PdfReader(file_pdf)

    semua_teks = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            semua_teks += "\n" + text

    lines = [line.strip() for line in semua_teks.split("\n") if line.strip()]

    month_map = {
        "JAN": "01", "FEB": "02", "MAR": "03",
        "APR": "04", "MAY": "05", "JUN": "06",
        "JUL": "07", "AUG": "08", "SEP": "09",
        "OCT": "10", "NOV": "11", "DEC": "12"
    }

    tanggal_pattern = re.compile(
        r"^(\d{2})\s+(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\b",
        re.IGNORECASE
    )

    nominal_pattern = re.compile(
        r"^\d{1,3}(?:,\d{3})*$"
    )

    data = []
    i = 0

    while i < len(lines):

        match = tanggal_pattern.match(lines[i])

        if match:

            day, month = match.groups()

            transaksi = {
                "Tanggal": f"2026-{month_map[month.upper()]}-{day}",
                "Keterangan": "",
                "Nominal": 0
            }

            keterangan = []

            sisa_baris = lines[i][match.end():].strip()

            if sisa_baris:
                keterangan.append(sisa_baris)

            j = i + 1

            while j < len(lines):

                if tanggal_pattern.match(lines[j]):
                    break

                if nominal_pattern.match(lines[j]):

                    transaksi["Nominal"] = int(
                        lines[j].replace(",", "")
                    )
                    break

                if "Bunga Tabungan" not in lines[j]:
                    keterangan.append(lines[j])

                j += 1

            transaksi["Keterangan"] = " ".join(keterangan).strip()

            if (
                transaksi["Nominal"] > 0
                and "Bunga Tabungan" not in transaksi["Keterangan"]
            ):
                data.append(transaksi)

            i = j

        else:
            i += 1

    return pd.DataFrame(data)
def buat_pdf_laporan(dataframe, df_ringkasan):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    story = []
    styles = getSampleStyleSheet()
    
    story.append(Paragraph("LAPORAN KEUANGAN", styles['Title']))
    story.append(Spacer(1, 12))
    
    # Tabel Ringkasan (Menggunakan nama kolom yang dinamis)
    story.append(Paragraph("<b>A. Ringkasan Pengeluaran</b>", styles['Heading2']))
    data_ringkasan = [list(df_ringkasan.columns)] # Header otomatis dari kolom
    for _, row in df_ringkasan.iterrows():
        # Memformat nominal jika kolomnya adalah 'Nominal'
        row_data = [str(row[col]) if col != 'Nominal' else f"Rp {row[col]:,.0f}" for col in df_ringkasan.columns]
        data_ringkasan.append(row_data)
    
    story.append(Table(data_ringkasan, hAlign='LEFT'))
    story.append(Spacer(1, 20))
    
    # Tabel Detail Mutasi
    story.append(Paragraph("<b>B. Rincian Mutasi</b>", styles['Heading2']))
    data_mutasi = [["Tanggal", "Keterangan", "Nominal", "Kategori"]]
    for _, row in dataframe.iterrows():
        data_mutasi.append([
            str(row['Tanggal']), 
            str(row['Keterangan']), 
            f"Rp {row['Nominal']:,.0f}", 
            str(row['Kategori_AI'])
        ])
    story.append(Table(data_mutasi, hAlign='LEFT', colWidths=[70, 200, 80, 100]))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# ==========================================
# 2. MACHINE LEARNING & UI
# ==========================================
@st.cache_resource
def latih_model_ai():
    df_train = pd.read_csv('dataset_training.csv')
    df_train = df_train.dropna(subset=['Teks_Mutasi', 'Kategori'])
    vec = TfidfVectorizer(lowercase=True)
    X = vec.fit_transform(df_train['Teks_Mutasi'].astype(str))
    model = MultinomialNB().fit(X, df_train['Kategori'])
    return vec, model

vectorizer, model_ai = latih_model_ai()

uploaded_files = st.sidebar.file_uploader("Upload CSV/PDF", type=["csv", "pdf"], accept_multiple_files=True)

if uploaded_files:
    all_dfs = []
    for file in uploaded_files:
        df = pd.read_csv(file) if file.name.endswith('.csv') else ekstrak_dari_pdf(file)
        if df is not None and not df.empty: all_dfs.append(df)

    if all_dfs:
        df_baru = pd.concat(all_dfs, ignore_index=True)
        # Prediksi AI
        X = vectorizer.transform(df_baru['Keterangan'].fillna('').astype(str))
        df_baru['Kategori_AI'] = model_ai.predict(X)
        
        # Mapping Kelompok
        df_baru['Kelompok_Besar'] = df_baru['Kategori_AI'].replace({
            'Makan Bulanan': 'Pengeluaran Pokok', 'Transportasi': 'Pengeluaran Pokok',
            'Kebutuhan Pokok Bulanan': 'Pengeluaran Pokok', 'UKT 1 Semester': 'Akademik',
            'SPP Bulanan Pondok': 'Akademik', 'Kebutuhan Lainnya': 'Lain-lain'
        })
        
        st.dataframe(df_baru, use_container_width=True)
        
        # Visualisasi
        col1, col2 = st.columns(2)
        ringkasan_besar = df_baru.groupby('Kelompok_Besar')['Nominal'].sum().reset_index()
        ringkasan_kecil = df_baru.groupby('Kategori_AI')['Nominal'].sum().reset_index()
        
        with col1:
            st.write("Tabel Kluster Besar")
            st.table(ringkasan_besar.style.format({'Nominal': 'Rp{:,.0f}'}))
        with col2:
            st.write("Tabel Kluster Kecil")
            st.table(ringkasan_kecil.style.format({'Nominal': 'Rp{:,.0f}'}))
            
        fig, ax = plt.subplots()
        ax.pie(ringkasan_besar['Nominal'], labels=ringkasan_besar['Kelompok_Besar'], autopct='%1.1f%%')
        st.pyplot(fig)

        # Download Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_baru.to_excel(writer, index=False)
        
        st.download_button("⬇️ Download Excel", data=output.getvalue(), file_name="laporan.xlsx")
        
        # Download PDF
        pdf_data = buat_pdf_laporan(df_baru, ringkasan_besar)
        st.download_button("⬇️ Download PDF", data=pdf_data, file_name="laporan.pdf")