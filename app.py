import streamlit as st  #Import Required Libraries and Load Model
import pandas as pd
from joblib import load #for faster model loading and compatibility with scikit-learn

# Loading the trained model(SVM)
wholesalemodel = load('wholesalemodel.pkl')
#Importing Required Libraries and Loading selected Model

# Defining a function that takes several input parameters to predict whether it is Horeca or Retail
def predict_channel(region, fresh, milk, grocery, frozen, detergents_paper, delicassen):
    """Function to predict the channel (Horeca or Retail)"""
    # Map region input to numerical values
    region_mapping = {'Lisbon': 1, 'Oporto': 2, 'Other': 3} #The provided region input (e.g., "Lisbon") is converted into its corresponding 
    #numeric value using the dictionary.
    region_numeric = region_mapping[region]
#Machine learning models work with numeric inputs, not string labels. 
    #The numeric mapping enables the model to process and interpret region information.
    
    # Create input DataFrame
    inputs_dict = {
        'Region': [region_numeric],
        'Fresh': [fresh],
        'Milk': [milk],
        'Grocery': [grocery],
        'Frozen': [frozen],
        'Detergents_Paper': [detergents_paper],
        'Delicassen': [delicassen]
    }
    input_df = pd.DataFrame(inputs_dict)

    # Predict channel uses the trained machine learning model (wholesalemodel) to predict the channel (Horeca or Retail).
    channel = wholesalemodel.predict(input_df)[0]
    return channel

def app_layout():
    """Function to define the app layout"""
    st.title('Wholesale Customer Channel Prediction')
    st.header('Enter the details:')

    # Region input with a default selection
    region = st.radio('Region:', ['Lisbon', 'Oporto', 'Other'], index=0)

    # User prompt is styled for better visibility.
    st.markdown("**Enter or adjust the spend values using the +/- buttons:**")
    
    # Numerical inputs with default value 10 on UI
    fresh = st.number_input('Fresh (Annual Spend):', value=10)
    milk = st.number_input('Milk (Annual Spend):', value=10)
    grocery = st.number_input('Grocery (Annual Spend):', value=10)
    frozen = st.number_input('Frozen (Annual Spend):', value=10)
    detergents_paper = st.number_input('Detergents_Paper (Annual Spend):', value=10)
    delicassen = st.number_input('Delicassen (Annual Spend):', value=10)

    # Creates a placeholder for displaying the prediction.It allows dynamically updating content.
    prediction_placeholder = st.empty()

    # Prediction button
    if st.button('Predict Channel'):
        channel = predict_channel(region, fresh, milk, grocery, frozen, detergents_paper, delicassen)
        channel_name = 'Hotel/Restaurant/Cafe' if channel == 1 else 'Retail'
        
        # Displays the predicted channel in the output field using markdown for styling
        prediction_placeholder.markdown(
            f"<div style='font-size:20px; font-weight:bold; color:blue;'>"
            f"Predicted Channel: {channel_name} (Channel {channel})</div>",
            unsafe_allow_html=True,
        )

if __name__ == '__main__':
    app_layout()
