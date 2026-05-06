import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

# ─── Environment ──────────────────────────────────────────────────────────────
load_dotenv()

# ─── State ────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    # holds the full conversation history as a list of alternating human and AI messages
    # this is what gives the agent "memory" -- every invoke receives the entire history
    messages: List[Union[HumanMessage, AIMessage]]

# ─── Model ────────────────────────────────────────────────────────────────────
# initializes the local LLM using the model name defined in .env
# e.g. OLLAMA_MODEL=qwen2.5:3b
# Ollama must be running locally (ollama serve) before invoking the agent
llm = ChatOllama(model=os.getenv('OLLAMA_MODEL'))

# ─── Node ─────────────────────────────────────────────────────────────────────
def process(state: AgentState) -> AgentState:
    # passes the full conversation history to the LLM
    # the model sees all previous human and AI messages, enabling multi-turn conversation
    response = llm.invoke(state['messages'])
    
    # appends the AI response to the conversation history in state
    state['messages'].append(AIMessage(content = response.content))
    
    # prints the AI response and the full current state for debugging
    print(f'\nAI: {response.content}')
    print('CURRENT STATE: ', state['messages'])
    
    return state

# ─── Graph ────────────────────────────────────────────────────────────────────
graph = StateGraph(AgentState)

graph.add_node('process', process)
graph.add_edge(START, 'process')
graph.add_edge('process', END)

agent = graph.compile()

# ─── Conversation Loop ────────────────────────────────────────────────────────
# stores the full conversation history across multiple turns
# persists outside the graph so it survives between invocations
conversation_history = []

user_input = input('Enter: ')

# keeps the conversation running until the user types 'exit'
while user_input != 'exit':
    # wraps the raw user string into a HumanMessage and adds it to history
    conversation_history.append(HumanMessage(content = user_input))
    
    # invokes the graph with the full conversation history
    # the agent sees all previous messages, not just the latest one
    result = agent.invoke({'messages': conversation_history})

    # updates the local history with the result from the graph
    # result['messages'] includes all previous messages + the new AI response
    conversation_history = result['messages']

    user_input = input('Enter: ')

# ─── Logging ──────────────────────────────────────────────────────────────────
# writes the full conversation to a text file after the session ends
# utf-8 encoding is required to handle special characters in AI responses
with open('logging.txt', 'w', encoding='utf-8') as file:
    file.write('Your conversation log:\n')
    for message in conversation_history:
        # isinstance() checks the type of each message to label it correctly
        if isinstance(message, HumanMessage):
            file.write(f'You: {message.content}\n')
        elif isinstance(message, AIMessage):
            file.write(f'AI: {message.content}\n\n')
    file.write('End of Conversation')

print('Conversation saved to logging.txt')