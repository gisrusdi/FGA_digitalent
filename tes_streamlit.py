import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Muat model
model = joblib.load('flight_model.pkl')

# Muat data untuk mendapatkan varian lain
df = pd.read_csv("Clean_Dataset.csv", usecols=lambda column: column != 'Unnamed: 0')

# Preprocessing fungsi untuk variabel input
def preprocess(df):
    df['class'] = df['class'].map({'Economy': 0, 'Business': 1})
    df['stops'] = pd.factorize(df['stops'])[0]
    categorical_cols = ['airline', 'flight', 'source_city', 'departure_time', 'arrival_time', 'destination_city']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

# Preprocessing data
df, label_encoders = preprocess(df)

# Fungsi untuk prediksi
def predict(model, input_data):
    return model.predict(input_data)

# Antarmuka pengguna Streamlit
st.title("Flight Price Estimation")

# Ambil nama kota untuk selectbox dari label_encoders
source_city_labels = label_encoders['source_city'].classes_
destination_city_labels = label_encoders['destination_city'].classes_

# Form input untuk pengguna
source_city = st.selectbox("Source City", source_city_labels)
destination_city = st.selectbox("Destination City", destination_city_labels)
days_left = st.slider("Days Left to Departure", min_value=int(df['days_left'].min()), max_value=int(df['days_left'].max()), value=10)

# Encode input kota ke angka
source_city_encoded = label_encoders['source_city'].transform([source_city])[0]
destination_city_encoded = label_encoders['destination_city'].transform([destination_city])[0]

# Filter data berdasarkan input pengguna
filtered_df = df[(df['source_city'] == source_city_encoded) & (df['destination_city'] == destination_city_encoded)]

# Batasi jumlah kombinasi dengan mengambil sampel
def generate_combinations(filtered_df, days_left, sample_size=10):
    airlines = filtered_df['airline'].unique()
    flights = filtered_df['flight'].unique()
    departure_times = filtered_df['departure_time'].unique()
    arrival_times = filtered_df['arrival_time'].unique()
    classes = [0, 1]  # 0: Economy, 1: Business
    stops = filtered_df['stops'].unique()
    durations = filtered_df['duration'].unique()

    # Ambil sampel dari setiap variabel untuk mengurangi jumlah kombinasi
    airlines = pd.Series(airlines).sample(min(sample_size, len(airlines)), replace=True).unique()
    flights = pd.Series(flights).sample(min(sample_size, len(flights)), replace=True).unique()
    departure_times = pd.Series(departure_times).sample(min(sample_size, len(departure_times)), replace=True).unique()
    arrival_times = pd.Series(arrival_times).sample(min(sample_size, len(arrival_times)), replace=True).unique()
    stops = pd.Series(stops).sample(min(sample_size, len(stops)), replace=True).unique()
    durations = pd.Series(durations).sample(min(sample_size, len(durations)), replace=True).unique()

    combinations = pd.DataFrame([
        [airline, flight, source_city_encoded, departure_time, stop, arrival_time, destination_city_encoded, class_type, duration, days_left]
        for airline in airlines
        for flight in flights
        for departure_time in departure_times
        for arrival_time in arrival_times
        for class_type in classes
        for stop in stops
        for duration in durations
    ], columns=['airline', 'flight', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class', 'duration', 'days_left'])

    return combinations

# Konversi input ke dataframe
input_data = generate_combinations(filtered_df, days_left)

# Fungsi untuk decoding hasil prediksi
def decode_results(df, label_encoders):
    for col, le in label_encoders.items():
        df[col] = le.inverse_transform(df[col])
    df['class'] = df['class'].map({0: 'Economy', 1: 'Business'})
    return df

# Tombol untuk melakukan prediksi
if st.button("Predict"):
    # Lakukan prediksi dalam batch
    batch_size = 1000
    predictions = []
    for i in range(0, len(input_data), batch_size):
        batch = input_data.iloc[i:i + batch_size]
        predictions.extend(predict(model, batch))

    input_data['predicted_price'] = predictions
    sorted_data = input_data.sort_values(by='predicted_price')

    # Decode hasil prediksi ke bentuk string
    decoded_data = decode_results(sorted_data.copy(), label_encoders)

    # Tampilkan hasil
    
    # Tampilkan 5 hasil termurah dari kelas ekonomi
    st.write("Top 5 Cheapest Economy Prices:")
    economy_data = decoded_data[decoded_data['class'] == 'Economy'].head(5)
    st.write(economy_data)

    # Tampilkan 5 hasil termurah dari kelas business
    st.write("Top 5 Cheapest Business Prices:")
    business_data = decoded_data[decoded_data['class'] == 'Business'].head(5)
    st.write(business_data)

    st.write("Estimated Flight Prices (sorted by cheapest):")
    st.write(decoded_data)