import streamlit as st
import pandas as pd
import numpy as np
import joblib
import chromadb
import json
import re
from groq import Groq
from pydantic import BaseModel

# Initialize fault history
if 'fault_history' not in st.session_state:
    st.session_state.fault_history = []

st.set_page_config(
    page_title="Sensor Fault Diagnosis System",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ AI-Powered Sensor Fault Diagnosis System")
st.markdown("---")


st.markdown("### 📊 Normal Operating Ranges")
col1,col2,col3=st.columns(3)
with col1:
    st.info("🌡️ Temperature\n\nNormal: 20–80°C\nFault: >90°C")
with col2:
    st.info("⚡ Voltage\n\nNormal: 3.0–5.5V\nFault: >6.0V or <2.0V")
with col3:
    st.info("🔌 Current\n\nNormal: 0.1–2.0A\nFault: <0.05A")
st.markdown("---")

@st.cache_resource
def load_model():
    return joblib.load('fault_model.pkl')

@st.cache_resource
def setup_chromadb():
    client = chromadb.Client()
    collection = client.create_collection("sensor_knowledge")
    documents = [
        "A voltage spike occurs when voltage exceeds normal operating range of 3.0-5.5V. This can damage sensitive components and microcontrollers. Immediate action: shut down the system and check the power supply unit.",
        "Temperature overload happens when sensor readings exceed 80 degrees celsius. This can cause component failure and circuit damage. Immediate action: shut down system, improve ventilation, check cooling systems.",
        "Current drop below 0.05A indicates a broken connection, loose wire, or short circuit. This is a critical fault. Immediate action: inspect all connections and replace damaged wires.",
        "Normal sensor operation requires temperature between 20-80°C, voltage between 3.0-5.5V, and current between 0.1-2.0A. Any reading outside these ranges indicates a potential fault.",
        "Voltage drop below 2.0V indicates power supply failure or battery depletion. The system may shut down unexpectedly. Immediate action: check power source and replace if necessary.",
        "Multiple simultaneous faults in temperature and voltage indicate a catastrophic system failure. Immediately shut down all systems and perform full diagnostic before restarting.",
        "Sensor calibration drift occurs when readings gradually shift from normal ranges over time. Regular calibration checks every 30 days are recommended for accurate readings."
    ]
    collection.add(
        documents=documents,
        ids=["doc1","doc2","doc3","doc4","doc5","doc6","doc7"]
    )
    return collection


import os

@st.cache_resource
def setup_groq():
    return Groq(api_key=os.environ.get("api_key"))


model = load_model()
collection = setup_chromadb()
groq_client = setup_groq()

st.sidebar.header("🔧 Sensor Input Values")
temperature = st.sidebar.slider("Temperature (°C)", 0.0, 150.0, 45.0)
voltage = st.sidebar.slider("Voltage (V)", 0.0, 10.0, 4.0)
current = st.sidebar.slider("Current (A)", 0.0, 3.0, 1.0)

if st.sidebar.button("🔍 Diagnose Fault"):

    new_reading = pd.DataFrame({
        'temperature': [temperature],
        'voltage': [voltage],
        'current': [current]
    })

    prediction = model.predict(new_reading)[0]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Temperature", f"{temperature}°C")
    with col2:
        st.metric("Voltage", f"{voltage}V")
    with col3:
        st.metric("Current", f"{current}A")

    st.markdown("---")

    if prediction == 1:
        st.error("⚠️ FAULT DETECTED")
        st.markdown("### 🤖 AI Diagnosis")

        with st.spinner("Analyzing fault..."):
            query = f"temperature is {temperature}°C, voltage is {voltage}V, current is {current}A. What faults are present and what should be done?"
            results = collection.query(query_texts=[query], n_results=2)
            context = "\n".join(results['documents'][0])

            prompt = f"""Analyze the fault sensor data and respond ONLY with a JSON object with these exact fields:
- fault_detected (bool)
- fault_reason (str)
- severity (str: low/medium/high/catastrophic)
- immediate_action (str)

Return only JSON. No explanation. No extra text.

Context:
{context}
"""
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            ai_response = response.choices[0].message.content

            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                class FaultDiagnosis(BaseModel):
                    fault_detected: bool
                    fault_reason: str
                    severity: str
                    immediate_action: str

                diagnosis = FaultDiagnosis(**data)

                st.markdown(f"**🔴 Fault Reason:** {diagnosis.fault_reason}")
                st.markdown(f"**⚠️ Severity:** {diagnosis.severity.upper()}")
                st.markdown(f"**✅ Immediate Action:** {diagnosis.immediate_action}")

                st.session_state.fault_history.append({'Temperature': f"{temperature}°C",
    'Voltage': f"{voltage}V",
    'Current': f"{current}A",
    'Fault Reason': diagnosis.fault_reason,
    'Severity': diagnosis.severity.upper(),
    'Action': diagnosis.immediate_action})
    else:
        st.success("✅ SYSTEM NORMAL")

if st.session_state.fault_history:
    st.markdown("---")
    st.markdown("### 📋 Fault History Log")
    history_df=pd.DataFrame(st.session_state.fault_history)
    st.dataframe(history_df,use_container_width=True)