from typing import List
import llama_index.core
from llama_index.core.agent import AgentRunner
from llama_index.core.tools import BaseTool
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec

from llama_index.llms.gpt4free import GPT4Free
import g4f

tools: List[BaseTool] = []
tools = DuckDuckGoSearchToolSpec().to_tool_list()

llama_index.core.set_global_handler("simple")

llm = GPT4Free(g4f_model=g4f.models.gpt_4)
agent = AgentRunner.from_llm(tools=tools, llm=llm, verbose=True)

while True:
    user_input = input(f"> ")
    if user_input.lower() in ['', 'exit', 'quit']:
        break

    response = agent.chat(user_input)
    print(f'{llm.get_provider()}>\n{response}', end="\n\n")
