import chainlit as cl
from typing import List
from ctransformers import AutoModelForCausalLM


llm = AutoModelForCausalLM.from_pretrained(
    "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
)


def get_prompt(instruction: str, history: List[str] = None) -> str:
    system = """You are an AI assistant that gives helpful answers.
                You answer the question in a short and concise way."""
    prompt = f"### System:\n{system}\n\n### User:\n"
    if history is not None:
        prompt += f"This is the conversation history: {''.join(history)}. Now answer the question: "
    prompt += f"{instruction}\n\n### Response:\n" 
    print(prompt)
    return prompt


def get_response(prompt: str) -> str:
    response = ""
    for word in llm(prompt, stream=True):
        response += word
    # now I would like
    
    return response


@cl.on_message
async def on_message(message: cl.Message):
    response = f"Hello, you just sent: {message.content}!"
    await cl.Message(response).send()
    
"""
history = []

question = "Which city is the capital of India?"

answer = ""
for word in llm(get_prompt(question), stream=True):
    print(word, end="", flush=True)
    answer += word
print()

history.append(answer)

question = "and which is of Belarus?"

for word in llm(get_prompt(question, history), stream=True):
    print(word, end="", flush=True)
print()
"""