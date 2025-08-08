print("Radhe Radhe")
import phi
from phi.agent import Agent
import phi.api
from phi.playground import Playground, serve_playground_app
from phi.model.groq import Groq
from groq import Groq as gr 
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os 
from dotenv import load_dotenv
 



load_dotenv()  # This loads variables from .env into os.environ

#llm agent 

client = gr(
    api_key=os.environ.get("GROQ_API_KEY")

)

'''chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of fast language models",
        }
    ],
    model="llama-3.3-70b-versatile",
    stream=False,
) 

print(chat_completion.choices[0].message.content)
'''

#web Search agent
web_search_agent = Agent(
    name = "Web search agent",
    role = "search the web for the information",
    provider = Groq(),
    tools = [DuckDuckGo()],
    instructions = ["Always include sources"],
    show_tools_calls = True,
    markdown = True,
)

## financial agent 
financial_agent = Agent(
    name = "Financial AI agent",
    role = "provide financial information and answer questions",
    provider = Groq(),
    tools = [
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals= True, company_news= True),
    ],
    instructions = ["Always include sources","use table to show data when possible"],
    show_tools_calls = True,
    markdown = True,
)

app = Playground(agents=[financial_agent, web_search_agent]).get_app()

if __name__=="__main__":
    serve_playground_app("finance_agent:app")