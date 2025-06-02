import streamlit as st
import numpy as np
import pandas as pd
import pickle

# =====================
# Cargar modelo y encoder
# =====================
with open('../models/best_xgb_final.pkl', 'rb') as f:
    model = pickle.load(f)

with open('../models/label_encoder_sub_grade.pkl', 'rb') as f:
    le_sub_grade = pickle.load(f)

# =====================
# Título de la App
# =====================
st.title("💳 Predicción de riesgo de crédito")
st.write("Ingrese los datos del solicitante o cargue un archivo CSV para evaluar el riesgo de incumplimiento de préstamo.")

# =====================
# Carga de CSV para predicciones en lote
# =====================
uploaded_file = st.file_uploader("Carga un archivo CSV con los datos de entrada", type=["csv"])

if uploaded_file is not None:
    df_csv = pd.read_csv(uploaded_file)
    st.write("✅ Archivo cargado correctamente. Mostrando las primeras filas:")
    st.dataframe(df_csv.head())

    # Predecir
    y_probs = model.predict_proba(df_csv)[:, 1]
    y_preds = model.predict(df_csv)
    df_csv['prob_default'] = y_probs
    df_csv['predicted_class'] = y_preds

    # Mostrar resultados
    st.subheader("📊 Resultados de las predicciones")
    st.dataframe(df_csv[['prob_default', 'predicted_class']].head())

    # Descargar resultados
    csv_out = df_csv.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar resultados como CSV", data=csv_out, file_name="predicciones_con_riesgo.csv", mime='text/csv')

else:
    # =====================
    # Inputs del usuario
    # =====================
    loan_amnt = st.number_input('Monto del préstamo (USD)', min_value=1000, max_value=50000, step=1000)
    term = st.selectbox('Plazo del préstamo (meses)', options=[36, 60])
    int_rate = st.number_input('Tasa de interés (%)', min_value=5.0, max_value=30.0, step=0.1, format="%.2f")
    sub_grade = st.selectbox('Sub-Grado crediticio', options=le_sub_grade.classes_)
    emp_length = st.slider('Años de experiencia laboral', min_value=0, max_value=10)
    home_ownership = st.selectbox('Tipo de tenencia de vivienda', options=['ANY','RENT', 'OWN', 'MORTGAGE'])
    annual_inc = st.number_input('Ingreso anual (USD)', min_value=10000, max_value=200000, step=1000)
    verifiaction_status = st.selectbox('Estado de verificación', options=['Not Verified','Source Verified', 'Verified'])
    issue_year = st.selectbox('Año de emisión', options=[2015, 2016])
    issue_month = st.selectbox('Mes de emisión', options=list(range(1, 13)))
    purpose = st.selectbox('Propósito del préstamo', options=['car','credit_card', 'debt_consolidation','home_improvement','house', 'major_purchase', 'medical', 'moving', 'other', 'renewable_energy', 'small_business', 'vacation'])
    earliest_cr_line_year = st.number_input('Año de la primera línea de crédito', min_value=1950, max_value=2016, step=1)
    earliest_cr_line_month = st.selectbox('Mes de la primera línea de crédito', options=list(range(1, 13)))
    open_acc = st.number_input('Cuentas abiertas', min_value=1, max_value=50, step=1)
    pub_rec = st.number_input('Registros públicos negativos', min_value=0, max_value=10, step=1)
    revol_bal = st.number_input('Crédito rotativo (USD)', min_value=0, max_value=100000, step=500)
    revol_util = st.number_input('Uso del crédito rotativo (%)', min_value=0.0, max_value=150.0, step=0.1)
    total_acc = st.number_input('Total de cuentas de crédito', min_value=0, max_value=100, step=1)
    initial_list_status = st.selectbox('Estado inicial del listado', options=['w', 'f'])
    mort_acc = st.number_input('Número de hipotecas', min_value=0, max_value=20, step=1)
    pub_rec_bankruptcies = st.number_input('Bancarrotas públicas', min_value=0, max_value=5, step=1) 
    zip_code = st.selectbox('Código Postal', options=['70466', '30723', '22690','48052','813','5113','29597','11650','93700','86630'])
    

    if st.button('Predecir Riesgo'):
        sub_grade_encoded = le_sub_grade.transform([sub_grade])[0]
        issue_d_scaled = issue_year + (issue_month - 1) / 12
        earliest_cr_line_scaled = earliest_cr_line_year + (earliest_cr_line_month - 1) / 12
        # Convertir tasa anual a mensual
        r = (int_rate / 100) / 12
        # Número de pagos
        n = term
        # Calcular el pago mensual
        installment = (loan_amnt * r) / (1 - (1 + r) ** -n)
        # Calcular ingreso mensual
        monthly_income = annual_inc / 12
        # Calcular DTI
        dti = (installment / monthly_income) * 100
        #densidad de credito
        densidad_crediticio = total_acc / open_acc

        input_data = pd.DataFrame([{
            'loan_amnt': loan_amnt,
            'term': term,
            'int_rate': int_rate,
            'installment': installment,
            'sub_grade': sub_grade_encoded,
            'emp_length': emp_length,
            'annual_inc': annual_inc,
            'dti': dti,
            'pub_rec': pub_rec,
            'revol_bal': revol_bal,
            'revol_util': revol_util,
            'initial_list_status': 0 if initial_list_status == 'w' else 1,
            'mort_acc': mort_acc,
            'pub_rec_bankruptcies': pub_rec_bankruptcies,
            'zip_code': int(zip_code),
            'earliest_cr_line_scaled': earliest_cr_line_scaled,
            'issue_d_scaled': issue_d_scaled,
            'densidad_crediticia': total_acc / open_acc,
            'home_ownership_ANY': int(home_ownership == 'ANY'),
            'home_ownership_MORTGAGE': int(home_ownership == 'MORTGAGE'),
            'home_ownership_OWN': int(home_ownership == 'OWN'),
            'home_ownership_RENT': int(home_ownership == 'RENT'),
            'verification_status_Not Verified': int(home_ownership == 'Not Verified'),
            'verification_status_Source Verified': int(home_ownership == 'Source Verified'),
            'verification_status_Verified': int(home_ownership == 'Verified'),
            'purpose_car': int(home_ownership == 'car'),
            'purpose_credit_card': int(home_ownership == 'credit_card'),
            'purpose_debt_consolidation': int(home_ownership == 'debt_consolidation'),
            'purpose_home_improvement': int(home_ownership == 'home_improvement'),
            'purpose_house': int(home_ownership == 'house'),
            'purpose_major_purchase': int(home_ownership == 'major_purchase'),
            'purpose_medical': int(home_ownership == 'medical'),
            'purpose_moving': int(home_ownership == 'moving'),
            'purpose_other': int(home_ownership == 'other'),
            'purpose_renewable_energy': int(home_ownership == 'renewable_energy'),
            'purpose_small_business': int(home_ownership == 'small_business'),
            'purpose_vacation': int(home_ownership == 'vacation')
        }])

        pred_prob = model.predict_proba(input_data)[0][1]
        pred_class = model.predict(input_data)[0]

        st.subheader("Resultado de la predicción")
        if pred_class == 1:
            st.error(f'🚫 Riesgo alto de incumplimiento ({pred_prob:.2%})')
        else:
            st.success(f'✅ Buen pagador esperado ({pred_prob:.2%})')