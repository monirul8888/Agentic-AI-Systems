import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import asyncio

load_dotenv()

SYSTEM_PROMPT = """
You are a helpful assistant with access to math and weather tools.

Available tools:
- add, subtract, multiply, divide: for any math calculations
- get_weather: for current weather in any city

Rules:
- Always use tools when the question involves math or weather.
- Never make up answers — always use the tools.
- Be concise and friendly in your responses.
"""


async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["mathServer.py"],
                "transport": "stdio"
            },
            "weather": {
                "url": "http://127.0.0.1:8000/mcp",
                "transport": "streamable_http"
            }
        }
    )

    tools = await client.get_tools()
    print("✅ Tools loaded:", [t.name for t in tools])

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)

    # Maintain conversation history
    conversation_history = []

    print("\n🤖 Assistant ready! Type 'quit' or 'exit' to stop.\n")
    print("─" * 50)

    while True:
        # Get user input
        try:
            user_input = input("\n👤 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "bye", "q"]:
            print("👋 Goodbye!")
            break

        # Add user message to history
        conversation_history.append(HumanMessage(content=user_input))

        print("⏳ Thinking...", end="\r")

        try:
            response = await asyncio.wait_for(
                agent.ainvoke({"messages": conversation_history}),
                timeout=30
            )

            # Extract assistant reply
            ai_message = response["messages"][-1]
            reply = ai_message.content

            # Update history with full response messages
            conversation_history = response["messages"]

            print(f"🤖 Assistant: {reply}")
            print("─" * 50)

        except asyncio.TimeoutError:
            print("❌ Request timed out. Please try again.")
            # Remove the last user message from history on timeout
            conversation_history.pop()

        except Exception as e:
            print(f"❌ Error: {type(e).__name__}: {e}")
            conversation_history.pop()


asyncio.run(main())