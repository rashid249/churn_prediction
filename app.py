import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model
@st.cache_resource
def load_model():
    with open('churn_model_pipeline.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Define the app
def main():
    st.title('Customer Churn Prediction')
    st.write('This app predicts whether a customer will churn based on their characteristics.')
    
    # Create input fields
    st.sidebar.header('Customer Information')
    
    # Personal information
    gender = st.sidebar.selectbox('Gender', ['Female', 'Male'])
    senior_citizen = st.sidebar.selectbox('Senior Citizen', ['No', 'Yes'])
    partner = st.sidebar.selectbox('Partner', ['No', 'Yes'])
    dependents = st.sidebar.selectbox('Dependents', ['No', 'Yes'])
    
    # Account information
    tenure = st.sidebar.slider('Tenure (months)', 0, 72, 12)
    contract = st.sidebar.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.sidebar.selectbox('Paperless Billing', ['No', 'Yes'])
    payment_method = st.sidebar.selectbox('Payment Method', [
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ])
    
    # Services
    phone_service = st.sidebar.selectbox('Phone Service', ['No', 'Yes'])
    multiple_lines = st.sidebar.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
    internet_service = st.sidebar.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    
    # Additional services
    online_security = st.sidebar.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
    online_backup = st.sidebar.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
    device_protection = st.sidebar.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
    tech_support = st.sidebar.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
    streaming_tv = st.sidebar.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
    streaming_movies = st.sidebar.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
    
    # Charges
    monthly_charges = st.sidebar.number_input('Monthly Charges ($)', 0.0, 200.0, 70.0)
    total_charges = st.sidebar.number_input('Total Charges ($)', 0.0, 10000.0, 1000.0)
    
    # Create a dictionary with the input data
    input_data = {
        'customerID': '0000-XXXXXX',  # Dummy value, not used in model
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    if st.sidebar.button('Predict Churn'):
        try:
            # Predict
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)
            
            # Display results
            st.subheader('Prediction Results')
            
            if prediction[0] == 1:
                st.error('This customer is likely to churn.')
            else:
                st.success('This customer is likely to stay.')
            
            st.write(f'Probability of churning: {prediction_proba[0][1]:.2%}')
            st.write(f'Probability of staying: {prediction_proba[0][0]:.2%}')
            
            # Show feature importance if available (for tree-based models)
            if hasattr(model.named_steps['model'], 'feature_importances_'):
                st.subheader('Feature Importance')
                try:
                    # Get feature names after preprocessing
                    feature_names = model.named_steps['preprocessing'].get_feature_names_out()
                    importances = model.named_steps['model'].feature_importances_
                    
                    # Create a DataFrame for visualization
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    st.bar_chart(importance_df.set_index('Feature').head(10))
                except Exception as e:
                    st.warning('Could not display feature importance.')
                    st.write(e)
            
        except Exception as e:
            st.error(f'An error occurred: {str(e)}')
    
    # Show raw input data if requested
    if st.checkbox('Show raw input data'):
        st.subheader('Raw Input Data')
        st.write(input_df)

if __name__ == '__main__':
    main()