import pandas as pd
from os import listdir
from os.path import isfile, join
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from transformers import TrainingArguments, Trainer
from datasets import Dataset

BASE_MODEL = "camembert-base"
LEARNING_RATE = 2e-5
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 20

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)


# Read files
PATH = "./semeval-2017-tweets_Subtask-A/downloaded/"
FILES = [PATH+f for f in listdir(PATH) if isfile(join(PATH, f))]
DFS = [pd.read_csv(file,sep="\t",names=['ID',"label",'text'],encoding="UTF-8") for file in FILES][:3]

raw_train_ds = Dataset.from_pandas(DFS[0])
raw_val_ds = Dataset.from_pandas(DFS[1])
raw_test_ds = Dataset.from_pandas(DFS[2])


# Preprocess
ds = {"train": raw_train_ds, "validation": raw_val_ds, "test": raw_test_ds}
yToFloat = {'positive':1.0,'neutral':0.0,'negative':-1.0}
def preprocess_function(examples):
    label = examples["label"] 
    examples = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)
    
    # Change this to real number
    examples["label"] = yToFloat[label]
    return examples

for split in ds:
    ds[split] = ds[split].map(preprocess_function, remove_columns=["ID"])


print(DFS[0])


# Metrics
def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)
    
    mse = mean_squared_error(labels, logits)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)
    single_squared_errors = ((logits - labels).flatten()**2).tolist()
    
    # Compute accuracy 
    # Based on the fact that the rounded score = true score only if |single_squared_errors| < 0.5
    accuracy = sum([1 for e in single_squared_errors if e < 0.25]) / len(single_squared_errors)
    
    return {"mse": mse, "mae": mae, "r2": r2, "accuracy": accuracy}




# Training
training_args = TrainingArguments(
    output_dir="../models/camembert-fine-tuned-regression-2",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    metric_for_best_model="accuracy",
    load_best_model_at_end=True,
    weight_decay=0.01,
)

import torch

class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs[0][:, 0]
        loss = torch.nn.functional.mse_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss
    

trainer = RegressionTrainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    compute_metrics=compute_metrics_for_regression,
)

trainer.train()

trainer.eval_dataset=ds["test"]
trainer.evaluate()

