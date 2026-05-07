# LangGraph Basic Agent Collection

This repository demonstrates several basic and intermediate agent patterns using [LangGraph](https://langchain-ai.github.io/langgraph/) and [Ollama](https://ollama.com/) for local LLMs. Each folder contains a progressively more advanced example, from a simple stateless agent to a document-editing agent with tool use.

## Project Overview

**Folders:**

- `1_SimpleBot/` — Minimal stateless agent: single-turn LLM call, no memory.
- `2_SimpleBot_Memory/` — Agent with memory: multi-turn conversation, logs chat to file.
- `3_ReAct_Agent/` — ReAct-style agent: uses tools (add, subtract, multiply), demonstrates tool-calling and reasoning.
- `4_Drafter/` — Document editing agent: uses tools to update and save documents, demonstrates tool use and stateful document editing.

See below for details and usage for each example.

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/) (for running local LLMs)

## Setup Instructions

### 1. Clone the Repository

```
git clone https://github.com/DozyXYZ/langgraph_agents_basic.git
cd langgraph_basic_agent
```

### 2. Install Ollama

Follow the instructions for your OS from the [Ollama installation page](https://ollama.com/download).

- **Windows:** Download and run the installer.
- **macOS:** `brew install ollama`
- **Linux:** Follow the instructions on the website.

### 3. Start Ollama

After installation, start the Ollama service:

```
ollama serve
```

### 4. Pull a Model

For example, to pull the Llama 3 model:

```
ollama pull llama3:8b
```

You can replace `llama3:8b` with any other supported model name.

### 5. Create a `.env` File

Create a `.env` file in the project root to store environment variables. Example:

```
OLLAMA_MODEL=llama3:8b
# Add other environment variables as needed
```

## Project Structure

```
langgraph_basic_agent/
│
├── 1_SimpleBot/
│   └── agent_bot.py
│
├── 2_SimpleBot_Memory/
│   ├── agent_bot.py
│   └── logging.txt
│
├── 3_ReAct_Agent/
│   └── react_agent.py
│
+├── 4_Drafter/
│   ├── drafter.py
│   └── document_update.txt
│
├── 5_RAG_agent/
│   ├── rag_agent.py
│   ├── Stock_Market_Performance_2024.pdf
│   ├── chroma.sqlite3
│   ├── rag_agent.PNG
│   └── README.md
│
└── README.md

### 5. RAG Agent (Retrieval-Augmented Generation)

Agent that answers questions about a PDF document using retrieval-augmented generation (RAG). Loads a PDF, splits it into chunks, stores embeddings in ChromaDB, and uses a retriever tool to answer questions with citations from the document.

```
python 5_RAG_agent/rag_agent.py
```

See `5_RAG_agent/README.md` for details, setup, and example questions.
```

## Usage

### 1. SimpleBot

Single-turn, stateless agent. Each input is processed independently.

```
python 1_SimpleBot/agent_bot.py
```

### 2. SimpleBot with Memory

Multi-turn conversational agent. Maintains conversation history and logs the chat to `logging.txt` after exit.

```
python 2_SimpleBot_Memory/agent_bot.py
```

### 3. ReAct Agent (Tool-using)

Agent that can use tools (add, subtract, multiply) and reason step-by-step. Demonstrates tool-calling and multi-step reasoning.

```
python 3_ReAct_Agent/react_agent.py
```

### 4. Drafter (Document Editing Agent)

Agent that helps update and save documents using tools. Demonstrates tool use, stateful editing, and saving to file.

```
python 4_Drafter/drafter.py
```

## References

- [Ollama Documentation](https://ollama.com/docs)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
