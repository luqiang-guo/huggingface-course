from transformers import AutoTokenizer

# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# # 单序列和多序列的api都一样
# sequence = "I've been waiting for a HuggingFace course my whole life."

# model_inputs = tokenizer(sequence)
# print(model_inputs)
# # {'input_ids': [101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
# sequences = [
#   "I've been waiting for a HuggingFace course my whole life.",
#   "So have I!"
# ]

# model_inputs = tokenizer(sequences)
# print(model_inputs)
# # {'input_ids': [[101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012, 102], [101, 2061, 2031, 1045, 999, 102]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]}


# #按照要求截断

# # 将序列填充到最大序列长度
# model_inputs = tokenizer(sequences, padding="longest")
# # print(model_inputs)
# # 将序列填充到模型最大长度
# # (512 for BERT or DistilBERT)
# model_inputs = tokenizer(sequences, padding="max_length")
# # print(model_inputs)
# # 对于小于max_length的填充
# model_inputs = tokenizer(sequences, padding="max_length", max_length=8)
# # print(model_inputs)

# sequences = [
#   "I've been waiting for a HuggingFace course my whole life.",
#   "So have I!"
# ]

# # 截断到模型最大输入长度
# # (512 for BERT or DistilBERT)
# model_inputs = tokenizer(sequences, truncation=True)
# # print(model_inputs)

# # 截断超过指定长度
# model_inputs = tokenizer(sequences, max_length=8, truncation=True)
# # print(model_inputs)
# # {'input_ids': [[101, 1045, 1005, 2310, 2042, 3403, 2005, 102], [101, 2061, 2031, 1045, 999, 102]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]}

# #还可以返回指定模型的数据类型
# sequences = [
#   "I've been waiting for a HuggingFace course my whole life.",
#   "So have I!"
# ]

# # Returns PyTorch tensors
# model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")
# # print(model_inputs)
# # Returns TensorFlow tensors
# model_inputs = tokenizer(sequences, padding=True, return_tensors="tf")
# # print(model_inputs)
# # Returns NumPy arrays
# model_inputs = tokenizer(sequences, padding=True, return_tensors="np")
# # print(model_inputs)


sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)
print(model_inputs["input_ids"])

#添加起始结尾标志位
tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

#
print(tokenizer.decode(model_inputs["input_ids"]))
print(tokenizer.decode(ids))
