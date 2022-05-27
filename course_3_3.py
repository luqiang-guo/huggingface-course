from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"

# 将原始数据转成训练所需要的数据
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
# 预处理数据集，并缓存到磁盘
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# 动态填充数据，每个batch 最大长度
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from transformers import TrainingArguments

# 获取参数
training_args = TrainingArguments("test-trainer")
print(training_args)

from transformers import AutoModelForSequenceClassification

# 实例化模型
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

from transformers import Trainer


trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
#训练
trainer.train()

# 监测
predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)