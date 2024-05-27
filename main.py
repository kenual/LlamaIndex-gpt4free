from llama_index.core.llms import ChatMessage
from llama_index.llms.gpt4free import GPT4Free

llm = GPT4Free(g4f_model="gpt-4")

messages = [
    ChatMessage(role="system", content="you are an English to French translator"),
    ChatMessage(role="user", content="hello there"),
    ChatMessage(role="assistant", content="Bonjour!"),
    ChatMessage(role="user", content="What is your name?"),
]
response = llm.chat(messages)
print(response)

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

