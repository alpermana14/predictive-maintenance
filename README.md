# Predictive Maintenance Copilot: Agentic AI & ML Forecasting

This project combines **Traditional Machine Learning** for time-series forecasting with **Agentic AI** to provide a "Copilot" experience for industrial maintenance. It predicts sensor failures on a conveyor belt and allows technicians to interact with the data through a natural language interface.

---

### 🚀 Features
* **Time-Series Forecasting:** Uses **LightGBM** to predict future trends for vibration (RMS), temperature, and current.
* **Anomaly Detection:** Implements a custom **IDK (Isolation Distribution Kernel)** algorithm to detect sensor irregularities.
* **Agentic AI:** A **LangGraph-powered** agent that can query machine manuals (PDF), search maintenance history, and draft work orders.
* **Real-time Scheduling:** An automated background scheduler that updates the machine state every 5 minutes from a MySQL database.

---

### 🛠️ Tech Stack
* **Backend:** Python (FastAPI), LangChain, LangGraph, LightGBM, Pandas, ChromaDB.
* **Frontend:** React.
* **Database:** MySQL (Sensor data) & Chroma (Vector storage).

---

### 📋 Prerequisites
Before running the project, ensure you have:
* **Python 3.10+**
* **Node.js** (for React frontend)
* **An OpenAI API Key**

---

### 🔧 Installation & Setup

#### 1. Backend Setup
```bash
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt
```

#### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### 🧠 How the Agent Works
The AI Copilot uses a Dynamic System Prompt that receives live machine context (current vibration, ISO 10816 zones, and forecast trends). It can execute three primary tools:

* **Manual Retriever:** Searches the Maintenance_Conveyor.pdf for technical specifications.
* **History Search:** Queries past maintenance logs stored in ChromaDB.
* **Work Order Drafter:** Automatically populates maintenance drafts based on detected anomalies.
