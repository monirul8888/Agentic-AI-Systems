from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from langgraph.graph.message import add_messages

from langgraph.prebuilt import (
    ToolNode,
    tools_condition
)

from langgraph.checkpoint.memory import MemorySaver


# =========================================
# LLM
# =========================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)


# =========================================
# STATE
# =========================================

class State(TypedDict):
    messages: Annotated[list, add_messages]


# =========================================
# GRAPH BUILDER
# =========================================

graph_builder = StateGraph(State)

memory = MemorySaver()


# =========================================
# TOOLS
# =========================================

search_tool = TavilySearch(max_results=2)

tools = [search_tool]


# =========================================
# BIND TOOLS
# =========================================

llm_with_tools = llm.bind_tools(tools)


# =========================================
# SYSTEM PROMPT
# =========================================

system_message = SystemMessage(
    content="""
You are a helpful AI assistant.

Rules:
1. Use tools only when needed.
2. After using tools, provide final answer.
3. Never call tools repeatedly.
4. Keep answers concise and clear.
"""
)


# =========================================
# CHATBOT NODE
# =========================================

def chatbot(state: State):

    messages = [system_message] + state["messages"]

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response]
    }


# =========================================
# TOOL NODE
# =========================================

tool_node = ToolNode(tools)


# =========================================
# ADD NODES
# =========================================

graph_builder.add_node("chatbot", chatbot)

graph_builder.add_node("tools", tool_node)


# =========================================
# GRAPH FLOW
# =========================================

graph_builder.add_edge(START, "chatbot")

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition
)

graph_builder.add_edge("tools", END)


# =========================================
# COMPILE GRAPH
# =========================================

graph = graph_builder.compile(
    checkpointer=memory
)


# =========================================
# CONFIG
# =========================================

config = {
    "configurable": {
        "thread_id": "1"
    },
    "recursion_limit": 5
}


# =========================================
# END TO END CHATBOT LOOP
# =========================================

print("\n🤖 AI Chatbot Started")
print("Type 'exit' to stop.\n")


while True:

    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("\n👋 Chatbot Stopped")
        break

    events = graph.stream(
        {
            "messages": [
                HumanMessage(content=user_input)
            ]
        },
        config,
        stream_mode="values"
    )

    print("\n🤖 Assistant:\n")

    for event in events:

        if "messages" in event:

            last_message = event["messages"][-1]

            last_message.pretty_print()

    print("\n" + "=" * 60 + "\n")