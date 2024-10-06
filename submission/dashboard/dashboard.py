import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


st.set_page_config(page_title='Final Project',  layout='wide')

# Header
st.title(
    "Final Project Dicoding: Bike Sharing Dataset :sparkles:"
)

# Load Dataframe
df = pd.read_csv("submission/dashboard/main_data.csv")

m1, m2, m3= st.columns((1, 1, 1))
m1.metric(label= "Data yang dipakai :", value= "Hourly Basis")
m2.metric(label= "Total Data :", value = str(len(df)))
m3.write('')

# Menampilkan seluruh dataframe
d1, d2 = st.columns(2)
with d1:
    st.subheader("Dataframe setelah preprocessing")
    st.write(df)

with d2:
    st.markdown("""
            Setelah preprocessing dataset dipangkas menjadi 11 kolom diantaranya:
            1. `season`: Musim dari pengamatan (1: Musim Semi, 2: Musim Panas, 3: Musim Gugur, 4: Musim Dingin).
            2. `weekday`: Hari dalam seminggu (0: Minggu, 1: Senin, ..., 6: Sabtu).
            3. `workingday`: Apakah hari tersebut adalah hari kerja (0: Bukan, 1: Ya).
            4. `weathersit`: Situasi cuaca (1: Cerah, 2: Berawan, 3: Hujan ringan, 4: Hujan lebat).
            5. `temp`: Suhu normalisasi dalam skala 0-1.
            6. `hum`: Kelembaban dalam skala 0-1.
            7. `windspeed`: Kecepatan angin dalam skala 0-1.
            8. `casual`: Jumlah pengguna sepeda yang tidak terdaftar.
            9. `registered`: Jumlah pengguna sepeda yang terdaftar.
            10. `cnt`: Jumlah total pengguna sepeda (`casual` + `registered`).
            11. `datetime`: Penggabungan kolom `dteday` dan `hr`
            """)

# Menampilkan Hasil dekomposisi
DecomposeResult = seasonal_decompose(df["cnt"], period=30)
trend = DecomposeResult.trend
seasonal = DecomposeResult.seasonal
residual = DecomposeResult.resid

def plot_decomposition(original, trend, seasonal, residual):
    """
    Plot the components of a decomposition. Pass in the original univarate data which was decomposed, and then the
    3 resulting components, being the trend, seasonal and residual parts.
    """
    # Plot the results
    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot(4, 1, 1)
    ax.plot(original, label = 'Original')
    ax.legend(loc = 'best')
    ax = fig.add_subplot(4, 1, 2)
    ax.plot(trend, label = 'Trend')
    ax.legend(loc = 'best')
    ax = fig.add_subplot(4, 1, 3)
    ax.plot(seasonal, label = 'Seasonality')
    ax.legend(loc = 'best')
    ax = fig.add_subplot(4, 1, 4)
    ax.plot(residual, label = 'Residuals')
    ax.legend(loc = 'best')
    return

fig = plot_decomposition(df["cnt"], trend, seasonal, residual)

col1, col2 = st.columns((1, 1))
with col1:
    st.subheader("Hasil Dekomposisi")
    st.pyplot(fig)
with col2:
    st.markdown("""
            Hasil dekomposisi dari data `cnt` (jumlah total pengguna sepeda) menunjukkan beberapa komponen penting yang dapat membantu dalam memahami pola dan tren dalam data ini.
            1. *Trend*: Komponen trend menunjukkan arah umum dari data penggunaan sepeda selama periode waktu yang diamati. Hal ini membantu untuk memahami apakah penggunaan sepeda meningkat, menurun, atau tetap stabil dalam jangka panjang. Pada hasil decomposisi ini, komponen trend (`trend`) meningkat seiring waktu, menunjukkan peningkatan jangka panjang pada nilai deret waktu, yang selaras dengan data asli.
            2. *Seasonal*: Pada hasil dekomposisi ini, komponen seasonal (`seasonal`) terlihat hampir konstan dan mendekati nol, menunjukkan bahwa rangkaian waktu tersebut mungkin tidak menunjukkan musiman yang signifikan. Hal ini dapat menunjukkan tidak adanya pola berulang secara berkala.
            3. *Residual*: Komponen residual menunjukkan variasi acak atau noise dalam data yang tidak dapat dijelaskan oleh komponen trend atau seasonal. Berguna untuk mengidentifikasi anomali atau kejadian tak terduga dalam data. Pada hasil decomposisi ini, komponen residual (`residual`) menjelaskan bila data masih mengandung fluktuasi yang tidak dapat dijelaskan oleh musiman atau tren.
            """)
    
# Menampilkan Korelasi

st.subheader("Correlation")
st.image("submission/data/corr.png")
