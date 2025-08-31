# AI Agent from Scratch  

This project is a lightweight **AI research agent** built from scratch with **OpenAI API** and **LangChain**.  
It integrates external knowledge sources like **DuckDuckGo** and **Wikipedia**, and extends them with my own **custom tools** — including a text-saving utility that automatically writes research outputs to a `.txt` file.  

The goal of this project is to demonstrate how an agent can:  
- Take a user query  
- Decide which tools to call  
- Collect and summarize information  
- Persist the results for later use  

---

## 🚀 Usage  

### 1. Create and activate a virtual environment
```bash
python3 -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
.\venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the agent
```bash
python app.py
```