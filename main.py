from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

# Define a python class which will specify the type of content that 
# we want our LLM to generate
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatOpenAI(model="gpt-5-nano")

# response = llm.invoke("What is the meaning of life?")
# print(response)

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            you are a research assistant that will help generate a research paper.
            Answer the user query and use necessary tools and every tool provided to you. Save the results with the save to txt tool.
            Wrap the output in this format and provide no other test\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}")
    ]
).partial(format_instructions=parser.get_format_instructions())

# Tools are used by LLM/Agent 
tools = [search_tool, wiki_tool, save_tool]

agent = create_tool_calling_agent(
    llm = llm,
    prompt = prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
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
