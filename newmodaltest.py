# app.py

import modal

app = modal.App("openai-parallel-agents")

image = (
    modal.Image.debian_slim()
    .pip_install("openai>=1.30.0")
)

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("openai-secret")]
)
def ask_openai(question: str, worker_id: int):
    import os
    from openai import OpenAI

    print("API key exists:", os.environ.get("OPENAI_API_KEY") is not None)
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"]
    )

    response = client.responses.create(
        model="gpt-5",
        input=question,
    )

    return {
        "worker": worker_id,
        "response": response.output_text,
    }


@app.local_entrypoint()
def main():
    question = "What are the three most important principles of good software design?"

    jobs = [
        ask_openai.spawn(question, i)
        for i in range(10)
    ]

    results = [job.get() for job in jobs]

    for result in results:
        print(f"\n=== Worker {result['worker']} ===")
        print(result["response"])