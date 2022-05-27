from datasets import load_dataset


# MRPC 数据集  标定语义是否一样的数据集

# ~/.cache/huggingface/dataset 
raw_datasets = load_dataset("glue", "mrpc")

raw_train_dataset = raw_datasets["train"]
raw_train_dataset[0]

print(raw_datasets)

print(raw_train_dataset[:5])

# features: ['sentence1', 'sentence2', 'label', 'idx'],
# OrderedDict([('sentence1', ['Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .', "Yucaipa owned Dominick 's before selling the chain to Safeway in 1998 for $ 2.5 billion .", 'They had published an advertisement on the Internet on June 10 , offering the cargo for sale , he added .', 'Around 0335 GMT , Tab shares were up 19 cents , or 4.4 % , at A $ 4.56 , having earlier set a record high of A $ 4.57 .', 'The stock rose $ 2.11 , or about 11 percent , to close Friday at $ 21.51 on the New York Stock Exchange .']), ('sentence2', ['Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .', "Yucaipa bought Dominick 's in 1995 for $ 693 million and sold it to Safeway for $ 1.8 billion in 1998 .", "On June 10 , the ship 's owners had published an advertisement on the Internet , offering the explosives for sale .", 'Tab shares jumped 20 cents , or 4.6 % , to set a record closing high at A $ 4.57 .', 'PG & E Corp. shares jumped $ 1.63 or 8 percent to $ 21.03 on the New York Stock Exchange on Friday .']), ('label', [1, 0, 1, 0, 1]), ('idx', [0, 1, 2, 3, 4])])

# 获取label 对应的索引
print(raw_train_dataset.features)