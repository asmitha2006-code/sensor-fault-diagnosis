######  ⚡ AI-Powered Sensor Fault Diagnosis System

> Detects embedded sensor faults in real-time using Machine Learning and explains them using a Generative AI RAG pipeline.

---

## 🚀 What It Does

Traditional sensor monitoring requires manual inspection. This system automates fault detection and diagnosis entirely:

- **Detects** anomalies in sensor readings (temperature, voltage, current) using a trained Random Forest model
- **Explains** what went wrong using a RAG pipeline — retrieving relevant knowledge and generating grounded AI explanations
- **Validates** all AI outputs using Pydantic for structured, reliable data
- **Displays** everything on a live Streamlit dashboard with fault history logging

---

## 🧠 Tech Stack

| Layer | Technology |
|---|---|
| ML Model | scikit-learn — Random Forest |
| LLM | Groq API — LLaMA 3.1 8B |
| Vector Database | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Structured Output | Pydantic |
| Dashboard | Streamlit |
| Language | Python |

---

## 🏗️ System Architecture

Sensor Input (Temperature / Voltage / Current)
↓
Random Forest Classifier → Fault or Normal
↓ (if fault)
ChromaDB Semantic Search → Relevant Knowledge
↓
Groq LLaMA API → Grounded Explanation
↓
Pydantic Validation → Structured Output
↓
Streamlit Dashboard → Live Display + Fault Log

---

## ⚙️ How To Run

**1. Clone the repository**
```bash
git clone https://github.com/YOURUSERNAME/sensor-fault-diagnosis.git
cd sensor-fault-diagnosis
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Generate the ML model**
```bash
python generate_model.py
```

**4. Set your Groq API key**
```bash
export GROQ_API_KEY="your_key_here"
```
Get a free key at groq.com

**5. Run the app**
```bash
streamlit run app.py
```

---

## 📁 Project Structure

sensor-fault-diagnosis/
├── app.py                  # Main Streamlit dashboard
├── generate_model.py       # Script to train and save ML model
├── requirements.txt        # Dependencies
├── README.md              # Documentation
└── sensor_data.csv        # Generated sensor dataset

---

## 🔍 Fault Detection Logic

| Sensor | Normal Range | Fault Condition |

| Temperature | 20–80°C | > 90°C |
| Voltage | 3.0–5.5V | > 6.0V or < 2.0V |
| Current | 0.1–2.0A | < 0.05A |

---

## 📌 Key Concepts Demonstrated

- End-to-end ML pipeline — data generation, training, evaluation, deployment
- RAG architecture — chunking, embedding, retrieval, generation
- Structured AI outputs with schema validation
- Real-time web dashboard with session state management
- Domain-specific AI grounding using embedded systems knowledge

---

*Built with Python, scikit-learn, ChromaDB, Groq, Pydantic, and Streamlit*
