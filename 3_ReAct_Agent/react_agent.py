from typing import Annotated # Provide additional context without affecting the type itself
from typing import Sequence # Handle automatically the state updates for sequences such as by adding new messages to a chat history
from typing import TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage # The foundational class for all message types in LangGraph
from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
import os

# Reducer Function
# Rule that controls how updates from nodes are combined with the existing state.
# Tell us how to merge data into the current state
# Without it, updates would have replaced the existing value entirely!
from langgraph.graph.message import add_messages 

load_dotenv()

# ─── State ────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    # Annotated[Sequence[BaseMessage], add_messages] is a special LangGraph reducer
    # instead of overwriting messages on each node return, add_messages APPENDS new
    # messages to the existing list automatically
    # this means nodes only need to return NEW messages, not the full history
    messages: Annotated[Sequence[BaseMessage], add_messages]

# ─── Tools ────────────────────────────────────────────────────────────────────
# @tool decorator converts a regular function into a LangChain tool
# the docstring is critical -- the LLM reads it to decide when to use the tool
# the type hints (a: int, b: int) tell the LLM what arguments to pass
@tool
def add(a: int, b: int):
    """This is an addition function that adds 2 numbers together"""
    return a + b

@tool
def subtract(a: int, b: int):
    """Subtraction function"""
    return a - b

@tool
def multiply(a: int, b: int):
    """Multiply function"""
    return a * b

# list of all tools available to the agent
# this list is passed to both the model and the ToolNode
tools =[add, subtract, multiply]

# ─── Model ────────────────────────────────────────────────────────────────────
# .bind_tools() tells the model what tools are available
# when the model decides to use a tool, it returns a tool_call instead of plain text
# the tool_call contains the tool name and arguments to pass
model = ChatOllama(model = os.getenv('OLLAMA_MODEL')).bind_tools(tools)

# ─── Nodes ────────────────────────────────────────────────────────────────────
def model_call(state: AgentState) -> AgentState:
    # SystemMessage sets the behavior and persona of the agent
    # it is prepended to every invoke so the model always has context
    system_prompt = SystemMessage(content = 'You are my AI assistant, please answer my query to the best of your ability')
    
    # prepends the system prompt to the full conversation history before invoking
    # the model either responds with plain text OR a tool_call
    response = model.invoke([system_prompt] + state['messages'])

    # returns only the NEW message -- add_messages reducer handles appending it
    return {'messages': [response]}

def should_continue(state: AgentState):
    messages = state['messages']
    last_messages = messages[-1]
    if not last_messages.tool_calls:
        return 'end'
    else:
        return 'continue'

# ─── Graph ────────────────────────────────────────────────────────────────────    
graph = StateGraph(AgentState)

# registers the model call as a node
graph.add_node('our_agent', model_call)

# ToolNode automatically executes whatever tool_calls the model requested
# it looks up the function by name from the tools list and runs it
# then appends the tool result as a ToolMessage back into state
tool_node = ToolNode(tools=tools)
graph.add_node('tools', tool_node)

# entry point -- always start at the agent
graph.add_edge(START, 'our_agent')

# after the agent runs, the router decides where to go next
# 'continue' -> tools node to execute the requested tool calls
# 'end'      -> END to return the final response to the user
graph.add_conditional_edges(
    'our_agent',
    should_continue,
    {
        'continue': 'tools',
        'end': END
    }
)

# after tools execute, always go back to the agent
# this creates a loop: agent -> tools -> agent -> tools -> ... -> end
# the loop continues until the model produces a response with no tool_calls
graph.add_edge('tools', 'our_agent')

app = graph.compile()

# ─── Streaming ────────────────────────────────────────────────────────────────
def print_stream(stream):
    for s in stream:
        # gets the most recent message from each streamed chunk
        message = s['messages'][-1]

        # tuples appear when the input is passed as ('user', 'content') format
        # AIMessage, ToolMessage etc. have a .pretty_print() method for clean output
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


# ─── Invoke ───────────────────────────────────────────────────────────────────
# the query intentionally has multiple chained operations + an unrelated request
# this tests whether the agent can:
# 1. chain tool calls in the correct order (add -> multiply -> subtract)
# 2. handle a non-tool request (math joke) in the same message using plain text
inputs = {'messages': [('user', 'Add 36 + 44, then multiply the result by 3, then subtract the result by 1. Also tell me a math joke')]}

# stream_mode='values' emits the full state after every node execution
# this lets print_stream show the conversation evolving step by step
print_stream(app.stream(inputs, stream_mode='values'))