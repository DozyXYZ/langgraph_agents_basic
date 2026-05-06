# langgraph_basic_agent

A simple Python project demonstrating basic agent functionality using LangGraph. This guide will help you set up the environment, install dependencies, configure Ollama, pull a model, and set up environment variables.

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/) (for running local LLMs)

## Setup Instructions

### 1. Clone the Repository

```
git clone <repo-url>
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
1_SimpleBot/
    agent_bot.py
2_SimpleBot_Memory/
    agent_bot.py
    logging.txt
```

## Usage

Run the desired agent bot script:

```
python 1_SimpleBot/agent_bot.py
# or
python 2_SimpleBot_Memory/agent_bot.py
```

## References

- [Ollama Documentation](https://ollama.com/docs)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

---
