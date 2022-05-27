from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader

# 加载数据
raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
# 预处理数据
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

# 预处理到文件
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# 数据加载
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 删除不需要的list  (参数是需要)。
tokenized_datasets = tokenized_datasets.remove_columns(
    ["sentence1", "sentence2", "idx"]
)
# #
# tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
# #
# tokenized_datasets.set_format("torch")
# #
# tokenized_datasets["train"].column_names
# #



# pytorch  dataloader
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

# 实例化 model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
# 实例化 optimizer
optimizer = AdamW(model.parameters(), lr=3e-5)
# 设置device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)

# lr_scheduler
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# 进度条
progress_bar = tqdm(range(num_training_steps))

# 训练
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

from datasets import load_metric

metric= load_metric("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()

#     metric.add_batch(
#         predictions=accelerator.gather(predictions),
#         references=accelerator.gather(batch["labels"]),
#     )
# metric.compute()
# # Use accelerator.print to print only on the main process.
# accelerator.print(f"epoch {epoch}:", metric)
