import os

import modal

app = modal.App()


@app.function(secrets=[modal.Secret.from_name("openai-secret")])
def f():
    print(os.environ["OPENAI_API_KEY"])
if __name__ == "__main__":
    with app.run():
        f()