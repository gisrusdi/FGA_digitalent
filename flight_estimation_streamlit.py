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

# Fungsi untuk decoding hasil prediksi
def decode_results(df, label_encoders):
    for col, le in label_encoders.items():
        df[col] = le.inverse_transform(df[col])
    df['class'] = df['class'].map({0: 'Economy', 1: 'Business'})
    return df

# Antarmuka pengguna Streamlit
st.title("Flight Price Estimation")

# Form input untuk pengguna
source_city = st.selectbox("Source City", label_encoders['source_city'].inverse_transform(df['source_city'].unique()))
destination_city = st.selectbox("Destination City", label_encoders['destination_city'].inverse_transform(df['destination_city'].unique()))
days_left = st.slider("Days Left to Departure", min_value=int(df['days_left'].min()), max_value=int(df['days_left'].max()), value=10)

# Filter data berdasarkan input pengguna
filtered_df = df[(df['source_city'] == label_encoders['source_city'].transform([source_city])[0]) & 
                 (df['destination_city'] == label_encoders['destination_city'].transform([destination_city])[0])]

# Buat semua kombinasi varian lainnya
def generate_combinations(filtered_df, days_left):
    airlines = filtered_df['airline'].unique()
    flights = filtered_df['flight'].unique()
    departure_times = filtered_df['departure_time'].unique()
    arrival_times = filtered_df['arrival_time'].unique()
    classes = [0, 1]  # 0: Economy, 1: Business
    stops = filtered_df['stops'].unique()
    durations = filtered_df['duration'].unique()
    
    combinations = pd.DataFrame([
        [airline, flight, label_encoders['source_city'].transform([source_city])[0], departure_time, stop, arrival_time, 
         label_encoders['destination_city'].transform([destination_city])[0], class_type, duration, days_left]
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

# Tombol untuk melakukan prediksi
if st.button("Predict"):
    predictions = predict(model, input_data)
    input_data['predicted_price'] = predictions
    sorted_data = input_data.sort_values(by='predicted_price')

    # Decode hasil prediksi ke bentuk string
    decoded_data = decode_results(sorted_data.copy(), label_encoders)
    
    st.write("Estimated Flight Prices (sorted by cheapest):")
    st.write(decoded_data)

    # Tampilkan 10 hasil termurah
    st.write("Top 10 Cheapest Prices:")
    st.write(decoded_data.head(10))
