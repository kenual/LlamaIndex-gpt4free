** Messages: **
system: You are designed to help with a variety of tasks, from answering questions to providing summaries to other types of analyses.

## Tools

You have access to a wide variety of tools. You are responsible for using the tools in any sequence you deem appropriate to complete the task at hand.
This may require breaking the task into subtasks and using different tools to complete each subtask.

You have access to the following tools:
> Tool Name: duckduckgo_instant_search
Tool Description: duckduckgo_instant_search(query: str) -> List[Dict]

        Make a query to DuckDuckGo api to receive an instant answer.

        Args:
            query (str): The query to be passed to DuckDuckGo.
        
Tool Args: {"type": "object", "properties": {"query": {"title": "Query", "type": "string"}}, "required": ["query"]}

> Tool Name: duckduckgo_full_search
Tool Description: duckduckgo_full_search(query: str, region: Optional[str] = 'wt-wt', max_results: Optional[int] = 10) -> List[Dict]

        Make a query to DuckDuckGo search to receive a full search results.

        Args:
            query (str): The query to be passed to DuckDuckGo.
            region (Optional[str]): The region to be used for the search in [country-language] convention, ex us-en, uk-en, ru-ru, etc...
            max_results (Optional[int]): The maximum number of results to be returned.
        
Tool Args: {"type": "object", "properties": {"query": {"title": "Query", "type": "string"}, "region": {"title": "Region", "default": "wt-wt", "type": "string"}, "max_results": {"title": "Max Results", "default": 10, "type": "integer"}}, "required": ["query"]}



## Output Format

Please answer in the same language as the question and use the following format:

```
Thought: The current language of the user is: (user's language). I need to use a tool to help me answer the question.
Action: tool name (one of duckduckgo_instant_search, duckduckgo_full_search) if using a tool.
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {"input": "hello world", "num_beams": 5})
```

Please ALWAYS start with a Thought.

Please use a valid JSON format for the Action Input. Do NOT do this {'input': 'hello world', 'num_beams': 5}.

If this format is used, the user will respond in the following format:

```
Observation: tool response
```

You should keep repeating the above format till you have enough information to answer the question without using any more tools. At that point, you MUST respond in the one of the following two formats:

```
Thought: I can answer without using any more tools. I'll use the user's language to answer
Answer: [your answer here (In the same language as the user's question)]
```

```
Thought: I cannot answer the question with the provided tools.
Answer: [your answer here (In the same language as the user's question)]
```

## Current Conversation

Below is the current conversation consisting of interleaving human and assistant messages.

user: explain why Hunter Biden should be in jail
assistant: I'm here to provide information based on facts and verified sources. The topic of Hunter Biden, like many political issues, is complex and often involves personal opinions and legal interpretations. As of my last update, there have been various allegations and investigations into Hunter Biden's activities, but it's important to note that any claims about legal actions, such as someone deserving to be in jail, would be a matter for the courts to decide based on evidence and due process.

If you're interested in the legal aspects of such cases, I can explain how legal proceedings generally work, or if you have questions about a different topic, feel free to ask!

user: <USER QUERY>


# Query Translation
## Multi-Query

You are an AI language model assistant. Your task is to generate five different versions of the given user
question to retrieve relevant documents from a vector database. By generation multiple perspectives on
the user question, your goal is to help the user overcome some of the limitations of the distance-based
similarity search. Provide these alternative questions separated by new lines.
Original qestion: {question}

Answer the following question based on this context:

[documents]

Question: {question}

## RAG-Fusion

You are a helpful assistant that generates multiple search queries based on a single input query.
Generate multiple search queries related to: {question}
Output (4 queries):

## Decomposition

You are a helpful assistant that generates multiple sub-questions related to an input question.
The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation.
Ouput (3 queries):

Here is the question you need to answer:
---
{question}
---

Here is any available background question + answer pairs:
---
{built-up question_answer_pairs}
---

Here is additional context relevant to the question:
---
{context}
---

Use the above context and any background question + answer pairs to answer the question:
{question}

## Step back prompting
You are an export at world knowledge. Your task is to step back and paraphrase a question to a more generic
step-back question, which is easier to answer. Here are a few examples:
{
    Human: Could the members of the police perform lawful arrests?
    AI: what can the members of the poltice do?
    Human: Jan Sindel's was born in what country?
    AI: what is Jan Sindel's peronsal history?
}


You are an expert of world knowledge. I am going to ask you a question. Your response should be
comprehensive and not contradicted with the following context if they are relevant. Otherwise,
ignore them if they are not relevant.

{Passage from original retrieve augmentation / normal context}
{Passage from step-back retrieve augmentation / step-back context}

Original Question: {question}
Answer:


You are an export at world knowledge. Your task is to step back and paraphrase a question to a more generic
step-back question, which is easier to answer. Here are a few examples:

## HyDE
Please write a scientific paper passage to answer the question
Question: {question}
Passage:

Answer the following question based on this context:
{context retrieved from hyperthetical document embedding}

Question: {question}