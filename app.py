import streamlit as st
import requests
import pandas as pd

TEMP_VALUE = 0.00001
GAMMA_0_VALUES = [0.01, 0.1, 0.5, 1.0, 10, 100, 1000000]

st.title("Quantum Visualization")

time_range = st.slider("Select Time Range", 0, 100, (0, 100), step=1)

gamma_0_selection = st.selectbox("Select Gamma Value", GAMMA_0_VALUES)

def generate_plots(temp_value, gamma_0_value, time_values):
    input_data = {
        "temp_value": temp_value,
        "gamma_0_values": [gamma_0_value],  
        "time_values": time_values
    }
    
    response = requests.post("http://127.0.0.1:8000/calculate", json=input_data)
    
    if response.status_code == 200:
        results = response.json()
        
        concurrence_results = results['concurrence']
        fidelity_results = results['state_fidelity']
        
        df_concurrence = pd.DataFrame(concurrence_results, index=time_values)
        df_fidelity = pd.DataFrame(fidelity_results, index=time_values)
        
        st.subheader("Concurrence vs Time")
        st.line_chart(df_concurrence)
        
        st.subheader("State Fidelity vs Time")
        st.line_chart(df_fidelity)
        
    else:
        st.error("Error generating plots. Please try again.")

generate_plots(TEMP_VALUE, gamma_0_selection, list(range(time_range[0], time_range[1] + 1)))
