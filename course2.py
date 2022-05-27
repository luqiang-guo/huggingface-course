# from transformers import pipeline
from transformers import AutoTokenizer
# classifier = pipeline("sentiment-analysis")
# classifier([
#     "I've been waiting for a HuggingFace course my whole life.", 
#     "I hate this so much!",
# ])


# 获取模型
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english" 
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# 将原始数据转换成模型输入数据
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.", 
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)

# from transformers import AutoTokenizer

# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)

from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)

outputs = model(**inputs)
print(outputs)