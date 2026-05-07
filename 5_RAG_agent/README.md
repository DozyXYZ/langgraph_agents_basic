# 5_RAG_agent — Retrieval-Augmented Generation Agent

This example demonstrates a Retrieval-Augmented Generation (RAG) agent using LangGraph, Ollama, and ChromaDB. The agent answers questions about the contents of a PDF document (Stock_Market_Performance_2024.pdf) by retrieving relevant information and citing sources.

## Features

- **PDF Loading:** Loads and splits a PDF into manageable text chunks.
- **Embeddings & Vector Store:** Uses Ollama embeddings and ChromaDB to store and search document chunks.
- **Retriever Tool:** Exposes a tool for the agent to search the document and return relevant passages.
- **RAG Workflow:** The agent uses the retriever tool to answer user questions, always citing the source text.
- **Interactive CLI:** Users can ask questions in a loop; type `exit` or `quit` to stop.
- **Architecture Diagram:** See `rag_agent.PNG` for a visual overview.

## Usage

1. **Install dependencies** (see project root for setup instructions).
2. **Set up environment variables** in `.env`:
   - `OLLAMA_MODEL` (e.g., `llama3:8b`)
   - `OLLAMA_EMBED_MODEL` (e.g., `nomic-embed-text`)
3. **Place your PDF** (e.g., `Stock_Market_Performance_2024.pdf`) in this folder.
4. **Run the agent:**

   ```bash
   python 5_RAG_agent/rag_agent.py
   ```

5. **Ask questions** about the PDF content. The agent will retrieve and cite relevant passages.

## Files

- `rag_agent.py` — Main RAG agent code
- `Stock_Market_Performance_2024.pdf` — Example PDF for retrieval
- `chroma.sqlite3` — ChromaDB vector store (auto-generated)
- `rag_agent.PNG` — Architecture diagram
- `.env` — Environment variables (not included; create your own)

## Example Questions

- "What was the best performing sector in 2024?"
- "Summarize the stock market trends in Q2."
- "List the key findings from the report."

## Notes

- The agent only answers based on the PDF content. If no relevant information is found, it will say so.
- You can replace the PDF with your own document for custom RAG tasks.

---

For more details, see the code in `rag_agent.py` and the architecture diagram.
