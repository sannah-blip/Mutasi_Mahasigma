import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import io

# Library untuk Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Library untuk membuat PDF resmi
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Konfigurasi Halaman Web Streamlit
st.set_page_config(page_title="Smart Financial Assistant", layout="wide")
st.title(" Smart Financial Assistant - Laporan Mutasi Mahasigma")
st.write("Aplikasi AI untuk mengelompokkan pengeluaran mutasi rekening secara otomatis untuk laporan semesteran.")
st.write("---")

# ==========================================
# 1. LOAD DATASET TRAINING SECARA DINAMIS
# ==========================================
@st.cache_resource
def latih_model_ai():
    df_train = pd.read_csv('dataset_training.csv')
    vectorizer = TfidfVectorizer(lowercase=True)
    X_train = vectorizer.fit_transform(df_train['Teks_Mutasi'])
    y_train = df_train['Kategori']
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return vectorizer, model

try:
    vectorizer, model_ai = latih_model_ai()
except FileNotFoundError:
    st.error(" Gagal memuat sistem! File 'dataset_training.csv' tidak ditemukan di folder project.")
    st.stop()


# ==========================================
# 2. FITUR UPLOAD FILE MUTASI BARU (INPUT)
# ==========================================
st.sidebar.header(" Unggah Data Mutasi")
uploaded_file = st.sidebar.file_uploader("Pilih file CSV Mutasi Rekening (Data Lapangan)", type=["csv"])

def deteksi_anomali(row):
    kategori = row['Kategori_AI']
    nominal = row['Nominal']
    if kategori == 'Makan Bulanan' and nominal > 50000:
        return 'PERINGATAN: Makan Terlalu Boros!'
    elif kategori == 'Kebutuhan Pokok Bulanan' and nominal > 300000:
        return 'PERINGATAN: Belanja Melebihi Batas!'
    return 'Aman'

# Fungsi generate PDF otomatis
def buat_pdf_laporan(dataframe, df_ringkasan):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    story = []
    
    styles = getSampleStyleSheet()
    # Gaya penulisan kostumisasi
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#1A5276'), spaceAfter=10)
    normal_style = styles['Normal']
    header_table_style = ParagraphStyle('HeaderStyle', parent=styles['Normal'], textColor=colors.white, bold=True)
    
    # 1. Judul Laporan
    story.append(Paragraph("LAPORAN PERTANGGUNGJAWABAN KEUANGAN BEASISWA", title_style))
    story.append(Paragraph("Dibuat otomatis oleh: Smart Financial Assistant AI", normal_style))
    story.append(Spacer(1, 15))
    
    # 2. Ringkasan Total per Kategori
    story.append(Paragraph("<b>A. Ringkasan Pengeluaran per Kategori</b>", styles['Heading2']))
    story.append(Spacer(1, 5))
    
    data_ringkasan = [[Paragraph("<b>Kategori</b>", header_table_style), Paragraph("<b>Total Pengeluaran</b>", header_table_style)]]
    for _, row in df_ringkasan.iterrows():
        data_ringkasan.append([row['Kategori'], f"Rp {row['Total Pengeluaran']:,.0f}"])
        
    t_ringkasan = Table(data_ringkasan, colWidths=[250, 150])
    t_ringkasan.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (1,0), colors.HexColor('#1A5276')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F2F4F4')])
    ]))
    story.append(t_ringkasan)
    story.append(Spacer(1, 20))
    
    # 3. Detail Transaksi Mutasi
    story.append(Paragraph("<b>B. Rincian Mutasi Rekening Terklasifikasi</b>", styles['Heading2']))
    story.append(Spacer(1, 5))
    
    data_mutasi = [[
        Paragraph("<b>Tanggal</b>", header_table_style), 
        Paragraph("<b>Keterangan Mutasi</b>", header_table_style), 
        Paragraph("<b>Nominal</b>", header_table_style), 
        Paragraph("<b>Kategori AI</b>", header_table_style)
    ]]
    for _, row in dataframe.iterrows():
        data_mutasi.append([
            row['Tanggal'], 
            Paragraph(row['Keterangan'], normal_style), 
            f"Rp {row['Nominal']:,.0f}", 
            row['Kategori_AI']
        ])
        
    t_mutasi = Table(data_mutasi, colWidths=[80, 180, 100, 140])
    t_mutasi.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2E4053')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#FBFCFC')])
    ]))
    story.append(t_mutasi)
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# ==========================================
# 3. PROSES DATA JIKA FILE SUDAH DI-UPLOAD
# ==========================================
if uploaded_file is not None:
    df_baru = pd.read_csv(uploaded_file)
    required_columns = ['Tanggal', 'Keterangan', 'Nominal']
    
    if not all(col in df_baru.columns for col in required_columns):
        st.error(f"Format kolom file salah! Pastikan file CSV memiliki kolom: {', '.join(required_columns)}")
    else:
        X_baru = vectorizer.transform(df_baru['Keterangan'].astype(str))
        df_baru['Kategori_AI'] = model_ai.predict(X_baru)
        df_baru['Status_Keuangan'] = df_baru.apply(deteksi_anomali, axis=1)
        
        st.subheader(" Hasil Pengelompokan Otomatis Transaksi")
        st.dataframe(df_baru, use_container_width=True)
        st.write("---")
        
        # ==========================================
        # 4. VISUALISASI DATA & RINGKASAN LAPORAN
        # ==========================================
        col1, col2 = st.columns([1, 1])
        ringkasan = df_baru.groupby('Kategori_AI')['Nominal'].sum()
        df_ringkasan = ringkasan.reset_index()
        df_ringkasan.columns = ['Kategori', 'Total Pengeluaran']
        
        with col1:
            st.subheader(" Grafik Proporsi Pengeluaran")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(ringkasan, labels=ringkasan.index, autopct='%1.1f%%', startangle=140,
                   colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0', '#d95f02'])
            center_circle = plt.Circle((0,0),0.70,fc='white')
            fig.gca().add_artist(center_circle)
            st.pyplot(fig)
            
        with col2:
            st.subheader(" Total Pengeluaran per Kategori")
            st.table(df_ringkasan.style.format({'Total Pengeluaran': 'Rp{:,.0f}'}))
            
            # ==========================================
            # 5. FITUR EKSPOR LAPORAN LANGSUNG KE PDF
            # ==========================================
            st.write("")
            st.subheader(" Ekspor Laporan Resmi")
            st.write("Klik tombol di bawah ini untuk mengunduh laporan berformat PDF resmi yang siap dikumpulkan.")
            
            # Membuat PDF menggunakan fungsi yang sudah didefinisikan
            pdf_data = buat_pdf_laporan(df_baru, df_ringkasan)
            
            # Tombol Download PDF Resmi
            st.download_button(
                label="Download Laporan Format PDF (.pdf)",
                data=pdf_data,
                file_name="Laporan_Resmi_Beasiswa.pdf",
                mime="application/pdf"
            )
else:
    st.info(" Silakan unggah file CSV mutasi rekening kelompokmu pada menu sidebar di sebelah kiri untuk memulai pengelompokan otomatis.")