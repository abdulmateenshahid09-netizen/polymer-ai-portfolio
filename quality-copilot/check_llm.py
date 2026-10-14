import time
import ollama

MODEL = "qwen2.5:7b"

start = time.time()
reply = ollama.chat(
    model=MODEL,
    messages=[{"role": "user", "content": "In one sentence, what is Cpk?"}],
    options={"temperature": 0},
)
print(reply["message"]["content"])
print("seconds:", round(time.time() - start, 1))
