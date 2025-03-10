
import sklearn
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the pre-trained pipeline and dataframe
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Laptop Price Predictor")

# Brand
company = st.selectbox('Brand', df['Company'].unique())

# Type of laptop (renamed to avoid reserved word conflicts)
laptop_type = st.selectbox('Type', df['TypeName'].unique())

# RAM (in GB)
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight of the laptop
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# Screen size in inches (used for calculating ppi)
screen_size = st.slider('Screen Size (in inches)', 10.0, 18.0, 13.0)

# Screen resolution
resolution = st.selectbox('Screen Resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160',
    '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'
])

# CPU (using the same input for both CPU name and brand)
cpu = st.selectbox('CPU', df['Cpu_brand'].unique())

# HDD (in GB)
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])

# SSD (in GB)
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])

# GPU
gpu = st.selectbox('GPU', df['Gpu_Brand'].unique())

# Operating System
os = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
    # Convert touchscreen and IPS to binary values
    touchscreen_bin = 1 if touchscreen == 'Yes' else 0
    ips_bin = 1 if ips == 'Yes' else 0

    # Calculate PPI (Pixels Per Inch)
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

    # Use the same value for both Cpu_name and Cpu_brand
    cpu_name = cpu
    cpu_brand = cpu

    # Create a DataFrame with the expected column names (must match training)
    query_df = pd.DataFrame({
        'Company': [company],
        'TypeName': [laptop_type],
        'Ram': [ram],
        'Weight': [weight],
        'Touch_screen': [touchscreen_bin],
        'Ips': [ips_bin],
        'ppi': [ppi],
        'Cpu_name': [cpu_name],
        'Cpu_brand': [cpu_brand],
        'HDD': [hdd],
        'SSD': [ssd],
        'Gpu_Brand': [gpu],
        'os': [os]
    })

    # Make prediction (assuming the model was trained on log(price))
    price_log = pipe.predict(query_df)[0]
    price = int(np.exp(price_log))

    st.title(f"The predicted price of this {company} Laptop is : {str(price)}/-" )
