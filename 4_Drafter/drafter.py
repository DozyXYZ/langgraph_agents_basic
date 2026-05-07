from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool

from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

import os

# ─── Environment ──────────────────────────────────────────────────────────────
load_dotenv()

# global variable that holds the current document content
# shared between the agent and tools via the 'global' keyword
document_content = ''

# ─── State ────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    # add_messages reducer automatically appends new messages instead of overwriting
    # nodes only need to return NEW messages, not the full history
    messages: Annotated[Sequence[BaseMessage], add_messages]

# ─── Tools ────────────────────────────────────────────────────────────────────
@tool
def update(content: str) -> str:
    """Updates the document with the provided content."""
    global document_content  # tells Python to modify the global variable, not create a local one
    document_content = content
    return f'Document has been updated successfully! The current content is:\n{document_content}'

@tool
def save(filename: str) -> str:
    """Save the current document to a text file and finish the process.
    
    Args:
        filename: Name for the text file
    """
    global document_content

    # ensures the file always has a .txt extension even if the user omits it
    if not filename.endswith('.txt'):
        filename = f'{filename}.txt'

    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f'\nDocument has been saved to: {filename}')
        return f'Document has been saved successfully to "{filename}"'
    except Exception as e:
        # returns the error as a string so the model can read and react to it
        # instead of crashing the whole program
        return f'Error saving document: {str(e)}'

tools = [update, save]

# ─── Model ────────────────────────────────────────────────────────────────────
# .bind_tools() makes the model aware of the update and save tools
# when the model decides to use a tool, it returns an AIMessage with tool_calls populated
model = ChatOllama(model=os.getenv('OLLAMA_MODEL')).bind_tools(tools)

# ─── Agent Node ───────────────────────────────────────────────────────────────
def our_agent(state: AgentState) -> AgentState:
    # system prompt is rebuilt every invocation so it always reflects
    # the latest document_content global variable
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """)

    # on the first run, state['messages'] is empty so we greet the user
    # on subsequent runs, we ask for the next action via input()
    if not state['messages']:
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input('\nWhat would you like to do with the document?: ')
        print(f'\nUSER: {user_input}')
        user_message = HumanMessage(content=user_input)

    # combines system prompt + full conversation history + new user message
    # gives the model full context of everything that happened so far
    all_messages = [system_prompt] + list(state['messages']) + [user_message]

    response = model.invoke(all_messages)

    print(f'\nAI: {response.content}')

    # hasattr check is a safety guard -- not all message types have tool_calls
    # prints which tools the model decided to call for visibility
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f'USING TOOLS: {[tc["name"] for tc in response.tool_calls]}')

    # returns the new user message and AI response as a list
    # add_messages reducer will append these to the existing history automatically
    # note: state['messages'] is included explicitly here because our_agent manages
    # its own input collection -- unlike a typical node that only returns new messages
    return {'messages': list(state['messages']) + [user_message, response]}

# ─── Router ───────────────────────────────────────────────────────────────────
def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation"""
    messages = state['messages']

    # if no messages exist yet, keep the graph running
    if not messages:
        return 'continue'

    # scans messages in reverse to find the most recent ToolMessage
    # a ToolMessage with 'saved' and 'document' in its content means
    # the save tool was successfully called -- time to end the session
    for message in reversed(messages):
        if (
            isinstance(message, ToolMessage) and
            'saved' in message.content.lower() and
            'document' in message.content.lower()
        ):
            return 'end'    # triggers the END edge, stopping the graph

    return 'continue'       # no save detected, keep the loop going

# ─── Print Helper ─────────────────────────────────────────────────────────────
def print_messages(messages):
    """Prints the last 3 messages, showing only ToolMessage results"""
    if not messages:
        return

    # only looks at the last 3 messages to avoid flooding the terminal
    # only prints ToolMessages -- AIMessage and HumanMessage are already
    # printed inside our_agent via print() statements
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f'TOOL RESULT: {message.content}')

# ─── Graph ────────────────────────────────────────────────────────────────────
graph = StateGraph(AgentState)

graph.add_node('agent', our_agent)
graph.add_node('tools', ToolNode(tools))  # ToolNode auto-executes tool_calls and wraps results in ToolMessage

# entry point -- always start at the agent node
graph.add_edge(START, 'agent')

# agent always routes to tools after responding
# note: this causes a crash if the model responds with no tool_calls
# a conditional edge from agent would be safer (see previous fix)
graph.add_edge('agent', 'tools')

# after tools execute, router decides whether to loop back or end
# 'continue' -> back to agent for the next user input
# 'end'      -> END when save tool has been successfully called
graph.add_conditional_edges(
    'tools',
    should_continue,
    {
        'continue': 'agent',
        'end': END
    }
)

app = graph.compile()

# ─── Entry Point ──────────────────────────────────────────────────────────────
def run_document_agent():
    print('\n ===== DRAFTER =====')

    # starts with an empty message history
    # the agent node handles the first greeting internally
    state = {'messages': []}

    # stream_mode='values' emits the full state after every node execution
    # this allows print_messages to show tool results as they happen
    for step in app.stream(state, stream_mode='values'):
        if 'messages' in step:
            print_messages(step['messages'])

    print('\n ===== DRAFTER FINISHED =====')

# standard Python entry point guard
# ensures run_document_agent() only runs when the file is executed directly
# and not when it is imported as a module by another file
if __name__ == '__main__':
    run_document_agent()