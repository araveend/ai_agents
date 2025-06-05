# ReviewAgentUI

A Streamlit-based application for document analysis and review, powered by embedding models and LangChain agents.

---

## 🚀 Features

- Upload and parse PDF files
- Generate and store document embeddings
- Perform intelligent Q&A over uploaded documents
- Conversational memory support
- Integrates with Ollama models and custom vector stores

---

## 📁 Project Structure

ReviewAgentUI/
├── app.py # Streamlit main app
├── agent_utils/
│ ├── init.py
│ ├── vector.py # Embedding and vector-related functions
│ └── memory.py # Memory setup for conversational agents
├── requirements.txt
└── README.md


---

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/araveend/ai_agents.git
cd ReviewAgentUI

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

## Run the application
streamlit run app.py
