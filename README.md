# README.md

# 🧠 Perpetual Reasoning Model

This project implements a **perpetually thinking reasoning agent** using LangGraph, enhanced with:

- **Energy-based Halting Conditions** (learned stopping)
- **Internal + External Memory RAG**
- **Full reasoning trace logging**
- **CLI + Streamlit loop controls**

---

## 🚀 Features

- 🧩 Modular LangGraph-based architecture
- 📚 Memory-aware RAG (Retrieval-Augmented Generation)
- 🧮 Halting logic via output entropy/energy
- 📓 Timestamped JSONL logs per reasoning step
- 🧵 Streamlit UI or CLI entrypoint
- 🔄 Iterative reasoning with context carry-over
- 💾 Memory retrieval and storage via FAISS
- 📊 Structured logging of each reasoning step
- ⚙️ Configurable via CLI

---

## 📁 Project Structure

```
perpetual_reasoner/
├── chains/
│   └── reasoning_chain.py        # Defines LLM reasoning chain
├── graphs/
│   └── reasoning_graph.py        # LangGraph for perpetual reasoning loop
├── halting/
│   └── energy_monitor.py         # Energy-based halting condition
├── memory/
│   └── memory_manager.py         # Internal + RAG memory system
├── utils/
│   └── logger.py                 # JSONL timestamped logger
├── main.py                       # CLI or Streamlit app entry
├── requirements.txt
└── README.md
```

---

## 🔧 Installation

```bash
git clone https://github.com/your-org/perpetual_reasoner.git
cd perpetual_reasoner
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🛠 Usage

### ▶ CLI Mode
```bash
python main.py --mode cli --max-steps 5 --stream False
```

### 🖼 Streamlit Mode
```bash
streamlit run main.py -- --mode streamlit
```

### Run Reasoning Loop

---
---

## 🧠 How It Works

1. **Context Initialization**
   - User provides a starting query
   - Memory + RAG fetch prior context

2. **Reasoning Iteration**
   - Chain runs and generates next output
   - Step is logged and memory updated

3. **Halting Check**
   - `EnergyMonitor` evaluates output entropy
   - Stops if energy < threshold

---

## 🧪 Extending

### 🔄 Custom Halting
Replace `energy_monitor.py` logic with:
- Reward-based RL
- Output perplexity
- Token entropy

### 📈 Live Monitoring
- Connect `logger.py` output to Superset/Streamlit dashboard

## 🧪 Future Extensions

- Add Streamlit dashboard
- Introduce multi-agent ReAct model
- Confidence-based halting criteria
- Memory pruning and summarization

---

## 📃 License
MIT License

---

## 🙋 Contributing
Pull requests welcome! Open issues or ideas for features.

---

## 🧩 Related Work
- LangGraph
- RAG Fusion
- Chain-of-Thought Iterative Models
- Neural Halting + ACT
