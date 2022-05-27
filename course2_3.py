# from transformers import BertTokenizer

# tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# print(tokenizer("Using a Transformer network is simple"))
# {'input_ids': [101, 7993, 170, 13809, 23763, 2443, 1110, 3014, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
# # tokenizer.save_pretrained( "directory_on_my_computer" )



from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)
print(tokens)
# ['Using', 'a', 'transform', '##er', 'network', 'is', 'simple']

ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
# [7993, 170, 11303, 1200, 2443, 1110, 3014]

decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string)
# Using a transformer network is simple
