from llama_index.llms.gpt4free import GPT4Free

llm = GPT4Free(g4f_model="gpt-4")

user_input = input(f"> ")
response = llm.stream_complete(user_input)

print()
for token in response:
    print(token.delta, end="")
print()
print(llm.get_provider())
