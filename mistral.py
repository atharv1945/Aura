import os
from llama_cpp import Llama
from config import Config

llm = Llama(
    model_path=Config.MISTRAL_MODEL_PATH,
    n_ctx=4096,
    n_threads=10
)

def ask_mistral(prompt, max_tokens=1024, temperature=0.2):
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        top_k=40,
        repeat_penalty=1.1
    )
    return output['choices'][0]['text']
