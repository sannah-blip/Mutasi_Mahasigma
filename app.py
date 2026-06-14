import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import io
import pypdf
import re
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
st.title("🎓 Smart Financial Assistant - Laporan Mutasi Mahasigma")
st.write("Aplikasi AI untuk mengelompokkan pengeluaran mutasi rekening secara otomatis untuk laporan semesteran.")
st.write("---")

# ==========================================
# 1. LOAD DATASET TRAINING SECARA DINAMIS
# ==========================================
@st.cache_resource
def latih_model_ai():
    # Load data
    df_train = pd.read_csv('dataset_training.csv') # Pastikan menggunakan dataset terbaru yang kita buat
    
    # PERBAIKAN ERROR NaN: Hapus baris kosong dan pastikan data berupa string
    df_train = df_train.dropna(subset=['Teks_Mutasi', 'Kategori'])
    df_train['Teks_Mutasi'] = df_train['Teks_Mutasi'].astype(str)

    # Proses TF-IDF
    vectorizer = TfidfVectorizer(lowercase=True)
    X_train = vectorizer.fit_transform(df_train['Teks_Mutasi'])
    y_train = df_train['Kategori']
    
    # Latih Model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return vectorizer, model

try:
    vectorizer, model_ai = latih_model_ai()
except FileNotFoundError:
    st.error("❌ Gagal memuat sistem! File dataset tidak ditemukan di folder project. Pastikan nama file di baris ke-28 sesuai.")
    st.stop()
def ekstrak_pdf_seabank(pdf_file):
    reader = pypdf.PdfReader(pdf_file)
    rows = []
    text = "\n".join([p.extract_text() or "" for p in reader.pages])
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    st.write(lines[:100])

    i = 0
    while i < len(lines):
        m = re.match(r"^(\d{2})\s+(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)", lines[i], re.I)
        if m:
            tanggal = lines[i]
            ket = []
            i += 1
            while i < len(lines):
                if re.match(r"^(\d{2})\s+(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)", lines[i], re.I):
                    break
                angka_rupiah = re.findall(
                    r'\b\d{1,3}(?:\.\d{3})+\b',
                    lines[i]
                )
                if len(angka_rupiah) == 2:

                    nominal = int(
                        angka_rupiah[0]
                        .replace('.', '')
                    )

                    saldo = int(
                        angka_rupiah[1]
                        .replace('.', '')
                    )

                    if nominal > 50000000:
                        continue
                    rows.append([
                        tanggal,
                        " ".join(ket),
                        nominal,
                        saldo
                    ])
        else:
            i += 1

    return pd.DataFrame(rows, columns=["Tanggal","Keterangan","Nominal","Saldo"])

# ==========================================
# 2. FITUR UPLOAD FILE MUTASI BARU (INPUT)
# ==========================================
st.sidebar.header("📂 Unggah Data Mutasi")
uploaded_file = st.sidebar.file_uploader("Pilih file CSV Mutasi Rekening (Data Lapangan)", type=["csv", "pdf"])

def load_data(file):

    if file.name.endswith('.csv'):

        df = pd.read_csv(file)

        return df

    elif file.name.endswith('.pdf'):

        df = ekstrak_pdf_seabank(file)

        return df

    return None

def deteksi_anomali(row):
    kategori = row['Kategori_AI']
    nominal = row['Nominal']
    if kategori == 'Makan Bulanan' and nominal > 20000:
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
    
    # Ringkasan Kelompok Besar
    story.append(Paragraph("<b>B. Ringkasan Kelompok Pengeluaran</b>", styles['Heading2']))
    
    ringkasan_kelompok = dataframe.groupby('Kelompok_Besar')['Nominal'].sum().reset_index()
    data_kelompok = [[
        Paragraph("<b>Kelompok</b>", header_table_style),
        Paragraph("<b>Total</b>", header_table_style)
    ]]

    for _, row in ringkasan_kelompok.iterrows():
        data_kelompok.append([
            row['Kelompok_Besar'],
            f"Rp {row['Nominal']:,.0f}"
        ])

    t_kelompok = Table(data_kelompok, colWidths=[250,150])
    t_kelompok.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1A5276')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('GRID',(0,0),(-1,-1),0.5,colors.black)
    ]))

    story.append(t_kelompok)
    story.append(Spacer(1,20))
    
    # 3. Detail Transaksi Mutasi
    story.append(Paragraph("<b>C. Rincian Mutasi Rekening Terklasifikasi</b>", styles['Heading2']))
    story.append(Spacer(1, 5))
    
    data_mutasi = [[
        Paragraph("<b>Tanggal</b>", header_table_style), 
        Paragraph("<b>Keterangan Mutasi</b>", header_table_style), 
        Paragraph("<b>Nominal</b>", header_table_style), 
        Paragraph("<b>Kategori AI</b>", header_table_style),
        Paragraph("<b>Kelompok</b>", header_table_style),
        Paragraph("<b>Status</b>", header_table_style),
    ]]
    
    for _, row in dataframe.iterrows():
        data_mutasi.append([
            str(row['Tanggal']),
            Paragraph(str(row['Keterangan']), normal_style),
            f"Rp {row['Nominal']:,.0f}",
            Paragraph(str(row['Kategori_AI']), normal_style),
            Paragraph(str(row['Kelompok_Besar']), normal_style),
            Paragraph(str(row['Status_Keuangan']), normal_style)
        ])
        
    t_mutasi = Table(data_mutasi, colWidths=[60, 140, 80, 90, 110, 120])
    t_mutasi.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2E4053')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#FBFCFC')])
    ]))
    story.append(t_mutasi)
    
    total_semua = dataframe['Nominal'].sum()
    story.append(Spacer(1,20))
    story.append(Paragraph(
        f"<b>Total Seluruh Pengeluaran : Rp {total_semua:,.0f}</b>",
        styles['Heading2']
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ==========================================
# 3. PROSES DATA JIKA FILE SUDAH DI-UPLOAD
# ==========================================
if uploaded_file is not None:

    df_baru = load_data(uploaded_file)
    st.subheader(
        "Preview Hasil Ekstraksi"
    )

    st.dataframe(
        df_baru.head(30)
    )
    csv_export = df_baru.to_csv(
        index=False
    ).encode('utf-8')
    st.download_button(
        label="Download CSV Hasil Konversi",
        data=csv_export,
        file_name="hasil_konversi_seabank.csv",
        mime="text/csv"
    )

    if df_baru is None:
        st.error("File tidak dapat dibaca")
        st.stop()

    required = [
        'Tanggal',
        'Keterangan',
        'Nominal'
    ]

    if not all(
        col in df_baru.columns
        for col in required
    ):
        st.error(
            "Format data tidak sesuai"
        )
        st.stop()

    X_baru = vectorizer.transform(
        df_baru['Keterangan']
        .fillna('')
        .astype(str)
    )

    df_baru['Kategori_AI'] = (
        model_ai.predict(X_baru)
    )
    required_columns = ['Tanggal', 'Keterangan', 'Nominal']

    if not all(col in df_baru.columns for col in required_columns):
        st.error(f"Format kolom file salah! Pastikan file CSV memiliki kolom: {', '.join(required_columns)}")
    else:
        # Prediksi dengan AI
        X_baru = vectorizer.transform(df_baru['Keterangan'].fillna('').astype(str))
        df_baru['Kategori_AI'] = model_ai.predict(X_baru)

        # Kelompok kategori besar
        df_baru['Kelompok_Besar'] = df_baru['Kategori_AI'].replace({
            'Makan Bulanan': 'Pengeluaran Pokok Bulanan',
            'Transportasi': 'Pengeluaran Pokok Bulanan',
            'Top Up eWallet': 'Pengeluaran Pokok Bulanan',
            'Belanja Online': 'Pengeluaran Pokok Bulanan',
            'Kebutuhan Pokok Bulanan': 'Pengeluaran Pokok Bulanan',
            'UKT 1 Semester': 'Pengeluaran Akademik',
            'SPP Bulanan Pondok': 'Pengeluaran Akademik',
            'Kebutuhan Lainnya': 'Lain-lain',
            'Tabungan Semesteran': 'Tabungan & Investasi',
        })

        # Deteksi Anomali
        df_baru['Status_Keuangan'] = df_baru.apply(deteksi_anomali, axis=1)
        
        st.subheader("📊 Hasil Pengelompokan Otomatis Transaksi")
        st.dataframe(df_baru, use_container_width=True)
        st.write("---")
        
        # ==========================================
        # 4. VISUALISASI DATA & RINGKASAN LAPORAN
        # ==========================================
        col1, col2 = st.columns([1, 1])
        
        # Ringkasan berdasarkan Kategori AI untuk Tabel
        ringkasan_kategori = df_baru.groupby('Kategori_AI')['Nominal'].sum().reset_index()
        ringkasan_kategori.columns = ['Kategori', 'Total Pengeluaran']
        
        # Ringkasan berdasarkan Kelompok Besar untuk Pie Chart
        ringkasan_kelompok = df_baru.groupby('Kelompok_Besar')['Nominal'].sum()
        
        with col1:
            st.subheader("📈 Grafik Proporsi Kelompok Pengeluaran")
            fig, ax = plt.subplots(figsize=(6, 6))
            wedges, texts, autotexts = ax.pie(
                ringkasan_kelompok,
                labels=None,
                autopct='%1.1f%%'
            )

            ax.legend(
                wedges,
                ringkasan_kelompok.index,
                loc='center left',
                bbox_to_anchor=(1,0.5)
            )
            center_circle = plt.Circle((0,0),0.70,fc='white')
            fig.gca().add_artist(center_circle)
            st.pyplot(fig)
            
        with col2:
            st.subheader("💰 Total Pengeluaran per Kategori")
            st.table(ringkasan_kategori.style.format({'Total Pengeluaran': 'Rp{:,.0f}'}))
            
        # ==========================================
        # 5. FITUR EKSPOR LAPORAN LANGSUNG KE PDF
        # ==========================================
        st.write("")
        st.subheader("📄 Ekspor Laporan Resmi")
        st.write("Klik tombol di bawah ini untuk mengunduh laporan berformat PDF resmi yang siap dikumpulkan.")
        
        # Membuat PDF
        pdf_data = buat_pdf_laporan(df_baru, ringkasan_kategori)
        
        # Tombol Download PDF Resmi
        st.download_button(
            label="⬇️ Download Laporan Format PDF (.pdf)",
            data=pdf_data,
            file_name="Laporan_Resmi_Beasiswa.pdf",
            mime="application/pdf"
        )
else:
    st.info("👈 Silakan unggah file CSV mutasi rekening lapangan pada menu sidebar di sebelah kiri untuk memulai.")