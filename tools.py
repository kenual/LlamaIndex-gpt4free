from typing import List
import llama_index.core
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import BaseTool
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec

from llama_index.llms.gpt4free import GPT4Free

tools: List[BaseTool] = []
tools = DuckDuckGoSearchToolSpec().to_tool_list()

llama_index.core.set_global_handler("simple")

llm = GPT4Free(g4f_model="gpt-4")
agent = ReActAgent.from_tools(tools=tools, llm=llm, verbose=True)
response = agent.chat("How to use ReActAgent with LlamaIndex?")
print(response, end="\n\n")