from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma

from operator import add as add_messages

from langgraph.graph import StateGraph, START, END

import os

load_dotenv()

# ─── Model ────────────────────────────────────────────────────────────────────
# temperature=0 makes the model deterministic -- it always picks the most likely token
# this minimizes hallucination which is critical when answering from a specific document
llm = ChatOllama(model=os.getenv('OLLAMA_MODEL'), temperature=0)

# embedding model converts text into numerical vectors for similarity search
# must be from the same family as the LLM for best compatibility
# e.g. OLLAMA_EMBED_MODEL=nomic-embed-text
embeddings = OllamaEmbeddings(model=os.getenv('OLLAMA_EMBED_MODEL'))

# ─── PDF Loading ──────────────────────────────────────────────────────────────
pdf_path = 'Stock_Market_Performance_2024.pdf'
# fails early with a clear message if the PDF is missing
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f'PDF file not found: {pdf_path}')

pdf_loader = PyPDFLoader(pdf_path)

# loads each page of the PDF as a separate Document object
# each Document has .page_content (text) and .metadata (page number, source, etc.)
try:
    pages = pdf_loader.load()
    print(f'PDF has been loaded and has {len(pages)} pages')
except Exception as e:
    print(f'Error loading PDF: {e}')
    raise

# ─── Chunking ─────────────────────────────────────────────────────────────────
# splits pages into smaller chunks so the retriever can find precise sections
# chunk_size=1000  -- each chunk is at most 1000 characters
# chunk_overlap=200 -- consecutive chunks share 200 characters to preserve context
#                      across chunk boundaries so ideas don't get cut off mid-sentence
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

pages_split = text_splitter.split_documents(pages)

# ─── Vector Store ─────────────────────────────────────────────────────────────
# persist_directory saves the vector store to disk so it survives between runs
# collection_name groups these documents under a named collection in ChromaDB
persist_directory = r'C:\Users\Kaido\Desktop\projects\langgraph_basic_agent\5_RAG_agent'
collection_name = 'stock_docs'

# creates the directory if it does not already exist
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    # converts each chunk into an embedding vector and stores them in ChromaDB
    # from_documents() handles both embedding generation and storage in one call
    vectorStore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f'Created ChromaDB vector store!')
except Exception as e:
    print(f'Error setting up ChromaDB: {str(e)}')
    raise

# ─── Retriever ────────────────────────────────────────────────────────────────
# as_retriever() wraps the vector store into a retriever interface
# search_type='similarity' uses cosine similarity to find the closest chunks
# k=5 returns the 5 most relevant chunks for each query
retriever = vectorStore.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 5}
)

# ─── Tool ─────────────────────────────────────────────────────────────────────
@tool
def retriever_tool(query: str) -> str:
    """This tool searches and returns the information from the Stock Market Performance 2024 document"""
    docs = retriever.invoke(query)

    # returns a clear message if no relevant chunks were found
    # prevents the model from hallucinating an answer when there is no data
    if not docs:
        return 'I found no relevant information in the Stock Market Performance 2024 document'

    # formats each retrieved chunk with a label so the model can cite them
    results = []
    for i, doc in enumerate(docs):
        results.append(f'Document {i+1}:\n{doc.page_content}')

    # joins all chunks with a blank line separator for readability
    return '\n\n'.join(results)

tools = [retriever_tool]

# binds the retriever tool to the LLM so it knows the tool is available
# overwrites the original llm variable with the tool-aware version
llm = llm.bind_tools(tools)

# ─── State ────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    # add_messages (imported as operator.add) appends new messages to the list
    # instead of overwriting -- nodes only need to return new messages
    messages: Annotated[Sequence[BaseMessage], add_messages]

# ─── Router ───────────────────────────────────────────────────────────────────
def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    # returns True if the model produced tool_calls, False if it gave a final answer
    # True  -> route to retriever_agent to execute the tool
    # False -> route to END and return the answer to the user
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

# ─── System Prompt ────────────────────────────────────────────────────────────
# defined outside the node so it is only created once, not on every invocation
# instructs the model to always use the retriever tool and cite its sources
system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

# ─── Tool Lookup ──────────────────────────────────────────────────────────────
# builds a dict mapping tool name -> tool object for fast lookup in take_action
# e.g. {'retriever_tool': <retriever_tool function>}
tools_dict = {our_tool.name: our_tool for our_tool in tools}

# ─── Nodes ────────────────────────────────────────────────────────────────────
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state"""
    messages = list(state['messages'])

    # prepends the system prompt on every call so the model always has instructions
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)

    # returns only the new AI message -- add_messages handles appending it to history
    return {'messages': [message]}

def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""
    # tool_calls is a list of dicts, each containing 'name', 'args', and 'id'
    tool_calls = state['messages'][-1].tool_calls
    results = []

    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")

        # guards against the model hallucinating a tool name that does not exist
        if not t['name'] in tools_dict:
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."

        else:
            # invokes the tool by looking it up in tools_dict using its name
            # .get('query', '') safely handles missing 'query' key in args
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")

        # wraps the tool result in a ToolMessage so the model can read it
        # tool_call_id links this result back to the specific tool_call that triggered it
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}

# ─── Graph ────────────────────────────────────────────────────────────────────
graph = StateGraph(AgentState)

graph.add_node('llm', call_llm)
graph.add_node('retriever_agent', take_action)

# always start at the LLM node
graph.add_edge(START, 'llm')

# after the LLM responds, router checks if tool calls are needed
# True  -> retriever_agent executes the tool calls
# False -> END returns the final answer to the user
graph.add_conditional_edges(
    'llm',
    should_continue,
    {
        True: 'retriever_agent',
        False: END
    }
)

# after tools execute, always go back to the LLM to process the results
# this creates the loop: llm -> retriever_agent -> llm -> ... -> END
graph.add_edge('retriever_agent', 'llm')

rag_agent = graph.compile()

# ─── Entry Point ──────────────────────────────────────────────────────────────
def running_agent():
    print('\n=== RAG AGENT ===')

    while True:
        user_input = input('\nWhat is your question: ')

        # accepts both 'exit' and 'quit' to stop the loop, case insensitive
        if user_input.lower() in ['exit', 'quit']:
            break

        # wraps the raw string into a HumanMessage before passing to the graph
        messages = [HumanMessage(content=user_input)]

        result = rag_agent.invoke({"messages": messages})

        print("\n=== ANSWER ===")
        # [-1] gets the final AIMessage after all tool calls have been resolved
        print(result['messages'][-1].content)

running_agent()