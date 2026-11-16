import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Sentiment Analysis C-Access",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* ===========================================================
MAIN CONTAINER
=========================================================== */

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    padding-left:3rem;
    padding-right:3rem;
}

/* ===========================================================
SIDEBAR
=========================================================== */

section[data-testid="stSidebar"]{
    border-right:1px solid #334155;
}

/* ===========================================================
METRIC CARD
=========================================================== */

[data-testid="metric-container"]{

    background-color:#1E293B;

    border:1px solid #334155;

    padding:20px;

    border-radius:15px;

    box-shadow:0px 5px 15px rgba(0,0,0,.25);

}

/* ===========================================================
BUTTON
=========================================================== */

.stButton>button{

    width:100%;

    height:50px;

    border-radius:12px;

    font-size:17px;

    font-weight:bold;

}

/* ===========================================================
DATAFRAME
=========================================================== */

[data-testid="stDataFrame"]{

    border-radius:12px;

    overflow:hidden;

}

/* ===========================================================
TEXT INPUT
=========================================================== */

.stTextInput>div>div>input{

    border-radius:10px;

}

/* ===========================================================
TEXT AREA
=========================================================== */

textarea{

    border-radius:12px;

}

/* ===========================================================
SELECTBOX
=========================================================== */

.stSelectbox{

    border-radius:10px;

}

/* ===========================================================
NUMBER INPUT
=========================================================== */

.stNumberInput{

    border-radius:10px;

}

/* ===========================================================
SUCCESS
=========================================================== */

.stSuccess{

    border-radius:12px;

}

/* ===========================================================
WARNING
=========================================================== */

.stWarning{

    border-radius:12px;

}

/* ===========================================================
INFO
=========================================================== */

.stInfo{

    border-radius:12px;

}

/* ===========================================================
ERROR
=========================================================== */

.stError{

    border-radius:12px;

}

/* ===========================================================
TITLE
=========================================================== */

h1{

    font-weight:800;

}

/* ===========================================================
SUBTITLE
=========================================================== */

h2,h3{

    font-weight:700;

}

</style>
""", unsafe_allow_html=True)

# ============================================================
# HOME
# ============================================================

st.title("🚆 Sentiment Analysis Dashboard")
st.markdown("---")

st.markdown("""
## Selamat Datang 👋

Dashboard ini digunakan untuk melakukan **Analisis Sentimen**
terhadap ulasan aplikasi **C-Access** menggunakan algoritma
**Naïve Bayes**.

### Fitur Dashboard

- 📥 Crawling Data Google Play Store
- 🧹 Preprocessing Data
- 🤖 Training Model Naïve Bayes
- 🔍 Prediksi Sentimen
- 📊 Visualisasi Data
- 📄 Dataset Viewer

Silakan pilih menu pada **sidebar** untuk memulai proses analisis.
""")

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("📥 **Crawling**\n\nMengambil review terbaru dari Google Play Store.")

with col2:
    st.warning("🧹 **Preprocessing**\n\nMembersihkan dan mempersiapkan data sebelum training.")

with col3:
    st.success("🤖 **Machine Learning**\n\nMelatih model Naïve Bayes dan melakukan prediksi sentimen.")