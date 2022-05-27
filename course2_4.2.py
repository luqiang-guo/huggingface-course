import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

# batched_ids = [
#   [ 200 , 200 , 200 ],
#   [ 200 , 200 ]
# ]

# padding_id = 100

# batched_ids = [
#   [200, 200, 200],
#   [200, 200, padding_id]
# ]

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [[200, 200, 200], [200, 200, tokenizer.pad_token_id]]

print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
print(model(torch.tensor(batched_ids)).logits)

# 结果不同

# tensor([[ 1.5694, -1.3895]], grad_fn=<AddmmBackward>)
# tensor([[ 0.5803, -0.4125]], grad_fn=<AddmmBackward>)
# tensor([[ 1.5694, -1.3895],
#         [ 1.3373, -1.2163]], grad_fn=<AddmmBackward>)


#添加掩码之后就可以得到一样的结果
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id]
]

attention_mask = [
  [1, 1, 1],
  [1, 1, 0]
]

outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print(outputs.logits)

# 序列很长的话要么使用更长序列的模型，或者截断输入序列。
# sequence = sequence[:max_sequence_length]