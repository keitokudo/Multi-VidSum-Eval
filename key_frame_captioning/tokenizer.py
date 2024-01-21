from transformers import AutoTokenizer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_tokenizer(config, annotation_file):
    if "bert_config" in config:
        tokenizer = AutoTokenizer.from_pretrained(config["bert_config"], do_lower_case=True)
        vocab = tokenizer.vocab
        pad_token_id = tokenizer.pad_token_id

        def bert_tokenizer(text):
            return tokenizer.encode(text, return_tensors='pt')[0]

    return bert_tokenizer, vocab, pad_token_id
