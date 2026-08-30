import os
from dotenv import load_dotenv
from openai import OpenAI

def main():
    load_dotenv()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("No OPENAI_API_KEY found in environment.")
        return

    # Print mask
    masked_key = key[:10] + "..." + key[-10:] if len(key) > 20 else key
    print(f"Testing key: {masked_key} (length={len(key)})")

    client = OpenAI(api_key=key)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=5
        )
        print("Success! Response from GPT-4o:", resp.choices[0].message.content)
    except Exception as e:
        print("Error type:", type(e).__name__)
        print("Error message:", str(e))

if __name__ == "__main__":
    main()
