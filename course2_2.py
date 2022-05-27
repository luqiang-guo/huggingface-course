# from transformers import BertConfig, BertModel

# # Building the config
# config = BertConfig()

# # Building the model from the config
# model = BertModel(config)

# print(config)

from transformers import BertModel



# https://huggingface.co/bert-base-cased  有很多训练好的模型
# 默认下载这里
# ~/.cache/huggingface/transformers
model = BertModel.from_pretrained("bert-base-cased")
# 模型保存
# model.save_pretrained("directory_on_my_computer")
# >>> ls directory_on_my_computer
# >>>config.json pytorch_model.bin

sequences = [
  "Hello!",
  "Cool.",
  "Nice!"
]

encoded_sequences = [
  [ 101, 7592,  999,  102],
  [ 101, 4658, 1012,  102],
  [ 101, 3835,  999,  102]
]

import torch
model_inputs = torch.tensor(encoded_sequences)
output = model(model_inputs)
print(output)