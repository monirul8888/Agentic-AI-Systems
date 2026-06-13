import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import asyncio

load_dotenv()

SYSTEM_PROMPT = """
You are a helpful assistant that uses tools to answer questions.
Always use the available tools. Return the exact tool output to the user.
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

    # Math query
    print("\n⏳ Running math query...")
    try:
        math_response = await asyncio.wait_for(
            agent.ainvoke({"messages": [("user", "What is 3 + 5?")]}),
            timeout=30
        )
        print("🧮 Math:", math_response["messages"][-1].content)
    except asyncio.TimeoutError:
        print("❌ Math timed out")
    except Exception as e:
        print(f"❌ Math error: {type(e).__name__}: {e}")

    # Weather query
    print("\n⏳ Running weather query...")
    try:
        weather_response = await asyncio.wait_for(
            agent.ainvoke({"messages": [("user", "What is the weather in Dhaka?")]}),
            timeout=30
        )
        print("🌦 Weather:", weather_response["messages"][-1].content)
    except asyncio.TimeoutError:
        print("❌ Weather timed out")
    except Exception as e:
        print(f"❌ Weather error: {type(e).__name__}: {e}")


asyncio.run(main())