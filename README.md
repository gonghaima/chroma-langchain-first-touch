# Step 1: Environment Setup

1. **Grab your OpenAI API Key**
2. **Set up Python application**
   - Create your project folder:
     ```bash
     mkdir chroma-langchain-demo
     ```
   - Change into the new directory and create your main Python file:
     ```bash
     cd chroma-langchain-demo
     touch main.py
     ```
   - (Optional) Create and activate your virtual environment:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
     _(On macOS, use `python3` instead of `python` for venv)_
   - Install OpenAI Python SDK:
     ```bash
     pip install openai
     ```

# Step 2: Install Chroma & LangChain

## Installing Chroma

Install ChromaDB using pip:

```bash
pip install chromadb
```

## Install LangChain, PyPDF, and tiktoken

Install these dependencies using pip:

```bash
pip install langchain pypdf tiktoken
```
