from transformers import AutoTokenizer
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
# tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])
# print(tokenized_sentences_1)
# print(tokenized_sentences_2)

inputs = tokenizer("This is the first sentence.", "This is the second one.")
# print(inputs)
# {'input_ids': [101, 2023, 2003, 1996, 2034, 6251, 1012, 102, 2023, 2003, 1996, 2117, 2028, 1012, 102], 
# 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], 
# 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}


# print(raw_datasets["train"]["sentence1"][:2])
# print(raw_datasets["train"]["sentence2"][:2])
# ['Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .', "Yucaipa owned Dominick 's before selling the chain to Safeway in 1998 for $ 2.5 billion ."]
# ['Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .', "Yucaipa bought Dominick 's in 1995 for $ 693 million and sold it to Safeway for $ 1.8 billion in 1998 ."]

# 返回的是一个字典，需要大量的内存空间
tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"][:2],
    raw_datasets["train"]["sentence2"][:2],
    padding=True,
    truncation=True,
)
# print(tokenized_dataset)
# {'input_ids': [[101, 2572, 3217, 5831, 5496, 2010, 2567, 1010, 3183, 2002, 2170, 1000, 
# 1996, 7409, 1000, 1010, 1997, 9969, 4487, 23809, 3436, 2010, 3350, 1012, 102, 7727, 2000, 
# 2032, 2004, 2069, 1000, 1996, 7409, 1000, 1010, 2572, 3217, 5831, 5496, 2010, 2567, 1997,
#  9969, 4487, 23809, 3436, 2010, 3350, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
#  [101, 9805, 3540, 11514, 2050, 3079, 11282, 2243, 1005, 1055, 2077, 4855, 1996, 4677, 
#  2000, 3647, 4576, 1999, 2687, 2005, 1002, 1016, 1012, 1019, 4551, 1012, 102, 9805, 3540, 11514, 
#  2050, 4149, 11282, 2243, 1005, 1055, 1999, 2786, 2005, 1002, 6353, 2509, 2454, 1998, 2853, 2009,
#   2000, 3647, 4576, 2005, 1002, 1015, 1012, 1022, 4551, 1999, 2687, 1012, 102]], 
#   'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
#   0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#   1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#   , 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
#   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], 
#   'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#     1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
#      1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}



def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


# 分批存储，填充值按照次分批的最大值  batched=True
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
# print(tokenized_datasets)
# DatasetDict({
#     train: Dataset({
#         features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
#         num_rows: 3668
#     })
#     validation: Dataset({
#         features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
#         num_rows: 408
#     })
#     test: Dataset({
#         features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
#         num_rows: 1725
#     })
# })


samples = tokenized_datasets["train"][:8]
samples = {
    k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]
}
# print(samples)
# print([len(x) for x in samples["input_ids"]])
# [50, 59, 47, 67, 59, 50, 62, 32]
# 最长67

from transformers import DataCollatorWithPadding

# 动态填充 到67
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

batch = data_collator(samples)
# print({k: v.shape for k, v in batch.items()})
# {'attention_mask': torch.Size([8, 67]),
#  'input_ids': torch.Size([8, 67]),
#  'token_type_ids': torch.Size([8, 67]),
#  'labels': torch.Size([8])}