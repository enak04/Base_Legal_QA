import os
import sys

# Set HF_HOME to the project's local cache folder so it gets bundled into the deploy image
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HOME"] = os.path.join(repo_root, ".hf_cache")

from transformers import BertTokenizer, BertModel

def main():
    model_name = "law-ai/InLegalBERT"
    print(f"Pre-downloading {model_name} to local cache: {os.environ['HF_HOME']}...")
    
    # Download tokenizer and model files
    BertTokenizer.from_pretrained(model_name)
    BertModel.from_pretrained(model_name)
    
    print("Download complete and cached locally!")

if __name__ == "__main__":
    main()
