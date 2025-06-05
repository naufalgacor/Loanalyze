import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Fungsi untuk memuat data dan melatih model
@st.cache_resource
def load_data_and_train_model():
    # Ganti dengan path dataset Anda
    df = pd.read_csv('loan_data_set.csv')
    
    # Fungsi untuk menghapus outlier
    def remove_outliers_iqr(_df, column):
        Q1 = _df[column].quantile(0.25)
        Q3 = _df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return _df[(_df[column] >= lower_bound) & (_df[column] <= upper_bound)]
    
    # Proses data
    numeric_column_to_clean = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
    for col in numeric_column_to_clean:
        df = remove_outliers_iqr(df, col)
    
    categorical_cols = ["Gender", "Married", "Dependents", "Self_Employed", "Credit_History"]
    for col in categorical_cols:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
    
    numeric_cols = ["LoanAmount", "Loan_Amount_Term"]
    for col in numeric_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    
    binary_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Loan_Status']
    le_dict = {}
    for col in binary_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    
    df = pd.get_dummies(df, columns=['Dependents', 'Property_Area'], drop_first=True)
    
    # Penskalaan fitur numerik
    numerik = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    scaler = StandardScaler()
    df[numerik] = scaler.fit_transform(df[numerik])
    
    # Feature engineering
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['IncomeToLoanRatio'] = df['TotalIncome'] / (df['LoanAmount'] + 1e-6)
    
    # Siapkan data training
    x = df.drop(["Loan_Status", "Loan_ID"], axis=1)
    y = df["Loan_Status"]
    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Latih model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(x_train, y_train)
    
    return model, scaler, le_dict, x.columns.tolist()

# Muat model dan komponen preprocessing
model, scaler, le_dict, all_columns = load_data_and_train_model()

# Fungsi untuk memproses input baru
def preprocess_input(input_data):
    # Konversi ke DataFrame
    df_input = pd.DataFrame([input_data])
    
    # Label Encoding untuk kolom biner
    binary_cols = ['Gender', 'Married', 'Education', 'Self_Employed']
    for col in binary_cols:
        le = le_dict[col]
        df_input[col] = le.transform([df_input[col].iloc[0]])[0]
    
    # One-hot encoding
    df_input = pd.get_dummies(df_input, columns=['Dependents', 'Property_Area'], drop_first=True)
    
    # Pastikan semua kolom ada
    for col in all_columns:
        if col not in df_input.columns:
            df_input[col] = 0
    
    # Urutkan kolom sesuai dengan data training
    df_input = df_input[all_columns]
    
    # Penskalaan fitur numerik
    numerik = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    df_input[numerik] = scaler.transform(df_input[numerik])
    
    # Feature engineering
    df_input['TotalIncome'] = df_input['ApplicantIncome'] + df_input['CoapplicantIncome']
    df_input['IncomeToLoanRatio'] = df_input['TotalIncome'] / (df_input['LoanAmount'] + 1e-6)
    
    return df_input

# Antarmuka Streamlit
st.title('Prediksi Persetujuan Pinjaman')
st.subheader('Masukkan detail peminjam:')

# Buat form input
with st.form("loan_form"):
    gender = st.selectbox('Gender', ['Male', 'Female'])
    married = st.selectbox('Married', ['Yes', 'No'])
    education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
    self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
    dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
    property_area = st.selectbox('Property Area', ['Urban', 'Semiurban', 'Rural'])
    applicant_income = st.number_input('Applicant Income', min_value=0)
    coapplicant_income = st.number_input('Coapplicant Income', min_value=0)
    loan_amount = st.number_input('Loan Amount', min_value=0)
    loan_amount_term = st.number_input('Loan Amount Term', min_value=0)
    credit_history = st.selectbox('Credit History', [1.0, 0.0])
    
    submit_button = st.form_submit_button("Prediksi")

# Ketika tombol submit ditekan
if submit_button:
    input_data = {
        'Gender': gender,
        'Married': married,
        'Education': education,
        'Self_Employed': self_employed,
        'Dependents': dependents,
        'Property_Area': property_area,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history
    }
    
    # Preprocessing input
    processed_input = preprocess_input(input_data)
    
    # Prediksi
    prediction = model.predict(processed_input)
    proba = model.predict_proba(processed_input)[0]
    
    # Tampilkan hasil
    status = "Disetujui" if prediction[0] == 1 else "Tidak Disetujui"
    color = "green" if prediction[0] == 1 else "red"
    
    st.subheader('Hasil Prediksi:')
    st.markdown(f"**Status Pinjaman:** <span style='color:{color};font-size:20px'>{status}</span>", 
                unsafe_allow_html=True)
    
    st.subheader('Probabilitas:')
    st.write(f"Tidak Disetujui: {proba[0]:.2%}")
    st.write(f"Disetujui: {proba[1]:.2%}")
    
    # Tampilkan data yang diproses (opsional)
    st.subheader('Data yang Diproses:')
    st.dataframe(processed_input)

# Tampilkan informasi dataset (opsional)
if st.checkbox('Tampilkan contoh data'):
    # Ganti dengan path dataset Anda
    sample_data = pd.read_csv('loan_data_set.csv').head(5)
    st.write(sample_data)