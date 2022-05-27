from datasets import load_dataset
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from datasets import load_metric
from accelerate import Accelerator , DistributedType
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
    get_scheduler,
    DataCollatorWithPadding,
)

MAX_GPU_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32

#
# - accelerator = Accelerator()

# 加载原始数据
raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-cased"
# 预处理数据
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
        return outputs

# 预处理到文件
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# 
tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["idx", "sentence1", "sentence2"],    # 删除不需要的列
)

# 修改为labels 
# 模型的
tokenized_datasets.rename_column_("label", "labels")

# If the batch size is too big we use gradient accumulation
gradient_accumulation_steps = 1
batch_size = 8

# def collate_fn(examples):
#     # On TPU it's best to pad everything to the same length or training will be very slow.
#     if accelerator.distributed_type == DistributedType.TPU:
#         return tokenizer.pad(examples, padding="max_length", max_length=128, return_tensors="pt")
#     return tokenizer.pad(examples, padding="longest", return_tensors="pt")



# # Instantiate dataloaders.
# train_dataloader = DataLoader(
#     tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size
# )
# eval_dataloader = DataLoader(
#     tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
# )

# 数据加载
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# pytorch  dataloader
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

seed = 34
set_seed(seed)

# Instantiate the model (we build the model here so that the seed also control new weights initialization)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, return_dict=True)

# 设置device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
# model.to(accelerator.device)

# Instantiate optimizer
optimizer = AdamW(params=model.parameters(), lr=2e-5, correct_bias=True)

# Prepare everything
# There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
# prepare method.
# model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
#     model, optimizer, train_dataloader, eval_dataloader
# )

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)

# Instantiate learning rate scheduler after preparing the training dataloader as the prepare method
# may change its length.
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=len(train_dataloader) * num_epochs,
)



# 进度条
progress_bar = tqdm(range(num_training_steps))

# 训练
# Now we train the model
for epoch in range(num_epochs):
    model.train()
    for step, batch in enumerate(train_dataloader):
        # We could avoid this line since we set the accelerator with `device_placement=True`.
        batch.to(accelerator.device)
        outputs = model(**batch)
        loss = outputs.loss
        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)
        if step % gradient_accumulation_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        progress_bar.update(1)

