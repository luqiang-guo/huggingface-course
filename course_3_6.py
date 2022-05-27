from accelerate import Accelerator
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from tqdm.auto import tqdm

accelerator = Accelerator()

# 加载数据
raw_datasets = load_dataset("glue", "mrpc")

checkpoint = "bert-base-uncased"

# 预处理数据
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# 数据加载
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=3e-5)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
# 预处理到文件
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# pytorch  dataloader
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

train_dl, eval_dl, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dl)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dl:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)