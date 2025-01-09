# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configure the page
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the trained model, scaler and feature columns"""
    try:
        model = joblib.load('titanic_model.pkl')
        scaler = joblib.load('titanic_scaler.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        return model, scaler, feature_columns
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def create_feature_array(inputs, feature_columns):
    """Create a properly formatted feature array"""
    # Convert inputs to a pandas DataFrame with correct column order
    df = pd.DataFrame([inputs], columns=feature_columns)
    return df

def main():
    model, scaler, feature_columns = load_model()
    
    if model is None or scaler is None or feature_columns is None:
        st.error("Failed to load the model. Please check if all required files exist.")
        return

    st.title("🚢 Titanic Survival Prediction")
    st.write("""
        Enter passenger information to predict their probability of survival on the Titanic.
        This model is based on historical data from the tragic voyage.
    """)

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            pclass = st.selectbox('Passenger Class', options=[1, 2, 3])
            sex = st.selectbox('Gender', options=['male', 'female'])
            age = st.number_input('Age', min_value=0.0, max_value=100.0, value=25.0)
            sibsp = st.number_input('Siblings/Spouses Aboard', min_value=0, max_value=10, value=0)

        with col2:
            parch = st.number_input('Parents/Children Aboard', min_value=0, max_value=10, value=0)
            fare = st.number_input('Ticket Fare (in £)', min_value=0.0, max_value=1000.0, value=7.25)
            embarked = st.selectbox('Port of Embarkation', options=['S', 'C', 'Q'])

        submitted = st.form_submit_button("Predict Survival Chances")

    if submitted:
        try:
            # Create inputs dictionary with encoded values
            inputs = {
                'Pclass': pclass,
                'Sex': 1 if sex == 'female' else 0,
                'Age': age,
                'SibSp': sibsp,
                'Parch': parch,
                'Fare': fare,
                'Embarked': {'S': 0, 'C': 1, 'Q': 2}[embarked]
            }
            
            # Create feature array with correct format
            features_df = create_feature_array(inputs, feature_columns)
            
            # Scale features
            scaled_features = scaler.transform(features_df)
            
            # Make prediction
            survival_proba = model.predict_proba(scaled_features)[0, 1]
            
            # Display prediction
            if survival_proba >= 0.5:
                st.success(f"Survival Probability: {survival_proba:.1%}")
            else:
                st.error(f"Non-survival Probability: {1-survival_proba:.1%}")
            
            # Display feature importance
            st.subheader("Feature Impact Analysis")
            importance = pd.DataFrame({
                'Feature': feature_columns,
                'Value': features_df.iloc[0],
                'Coefficient': model.coef_[0]
            })
            importance['Impact'] = importance['Value'] * importance['Coefficient']
            importance = importance.sort_values('Impact', key=abs, ascending=False)
            st.dataframe(importance[['Feature', 'Impact']].round(3))

        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")
            st.info("Please check your input values and try again.")

if __name__ == "__main__":
    main()