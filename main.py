# Load_dotenv loads environment variables from .env
from dotenv import load_dotenv

# Basemodel is used to define structured data models (schemas with validation)
from pydantic import BaseModel

# LangChain’s wrapper for the OpenAI chat API
from langchain_openai import ChatOpenAI

# LangChain’s wrapper for Anthropic models (Claude) (isn't being used in this AI-Agent)
# from langchain_anthropic import ChatAnthropic 

# To build structured prompts
from langchain_core.prompts import ChatPromptTemplate

# Parser that makes the model output conform to a Pydantic model 
# (so the AI’s response is valid structured data, not free text).
from langchain_core.output_parsers import PydanticOutputParser

# Functions to create an agent (an LLM that can call tools) 
# and to wrap it with an executor that runs the reasoning loop.
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Custom tools
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

# Define a python class which will specify 
# a schema for the AI’s final output
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatOpenAI(model="gpt-5-nano")

# response = llm.invoke("What is the meaning of life?")
# print(response)

# Creates a parser that forces the model’s final output to match the ResearchResponse schema
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", # sets AI’s role/instructions.
            """
            you are a research assistant that will help generate a research paper.
            Answer the user query and use necessary tools and every tool provided to you. Save the results with the save to txt tool.
            Wrap the output in this format and provide no other test\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"), # placeholder for previous conversation.
        ("human", "{query}"), # the user’s query.
        ("placeholder", "{agent_scratchpad}") # where the agent writes its thought process/tool calls.
    ]
).partial(format_instructions=parser.get_format_instructions())
# injects the schema formatting instructions so the LLM knows exactly how to structure its output

# Tools are used by LLM/Agent 
tools = [search_tool, wiki_tool, save_tool]

# Creates an agent that uses your LLM + prompt + tools.
agent = create_tool_calling_agent(
    llm = llm,
    prompt = prompt,
    tools=tools
)

# Wraps the agent in an executor that manages the full loop (running the model, calling tools, feeding results back)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Asks the user for a research question in the terminal
query = input("What can i help you research?")
raw_response = agent_executor.invoke({"query" : query})

# print(raw_response)

# {'query': 'what is the capital of France?', 
# 'output': '{"topic": "Capital of France", 
# "           summary": "Paris is the capital of France.", 
#             "sources": ["Britannica: Paris is the capital and most populous city of France.", 
#             "Wikipedia: Paris is the capital city of France."], 
#             "tools_used": ["General knowledge"]}'}

try:
    str_resp = parser.parse(raw_response.get("output"))
    print(str_resp)
except Exception as e:
    print("Error parsing response ", e)
