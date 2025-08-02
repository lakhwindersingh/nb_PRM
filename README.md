# README.md

# 🧠 Perpetual Reasoning Model

This project implements a perpetual reasoning loop using LangChain, LangGraph, and FAISS, designed to simulate continuous logical thought with memory retention and logging.

## 📦 Features
- Iterative reasoning with context carry-over
- Memory retrieval and storage via FAISS
- Structured logging of each reasoning step
- Configurable via CLI
- Optional STOP mechanism for halting loop

---

## 🚀 Getting Started

### 1. Clone & Setup
```bash
git clone https://github.com/yourusername/perpetual-reasoning-model.git
cd perpetual-reasoning-model
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Set OpenAI Key
Create a `.env` file:
```env
OPENAI_API_KEY=your-key-here
```

### 3. Run Reasoning Loop
```bash
python main.py --context "Why do humans seek meaning in life?" --steps 10
```

---

## 🧱 Project Structure
```
project_root/
├── main.py
├── chains/
├── memory/
├── logger/
├── graphs/
├── prompts/
├── data/
└── requirements.txt
```

## 🧪 Future Extensions
- Add Streamlit dashboard
- Introduce multi-agent ReAct model
- Confidence-based halting criteria
- Memory pruning and summarization

---

## 📄 License
MIT License
