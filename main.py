from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool
from llama_index.llms.gpt4free import GPT4Free
from llama_index.core.agent import ReActAgent
import llama_index.core

llama_index.core.set_global_handler("simple")

llm = GPT4Free(g4f_model="gpt-4")

messages = [
    ChatMessage(role="system",
                content="you are an English to French translator"),
    ChatMessage(role="user", content="hello there"),
    ChatMessage(role="assistant", content="Bonjour!"),
    ChatMessage(role="user", content="What is your name?"),
]
response = llm.chat(messages)
print(response)

# define sample Tool


def smart_combine(a: int, b: int) -> int:
    """smart combine two integers and returns the resulting integer"""
    return a * b + a + b


multiply_tool = FunctionTool.from_defaults(fn=smart_combine)

# initialize ReAct agent
agent = ReActAgent.from_tools([multiply_tool], llm=llm, verbose=True)
response = agent.chat("What is the result of smart combining 123 and 321 ?")
print(response, end="\n\n")

while True:
    user_input = input(f"> ")
    if user_input.lower() in ['', 'exit', 'quit']:
        break

    response = llm.stream_complete(user_input)

    print()
    for token in response:
        print(token.delta, end="")
    print(end="\n\n")
    print(llm.get_provider())
