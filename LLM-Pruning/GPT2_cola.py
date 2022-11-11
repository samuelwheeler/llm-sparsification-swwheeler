from transformers import pipeline, set_seed
from transformers import GPT2Tokenizer, GPT2Model, DataCollatorWithPadding,GPT2Config
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM, GPT2ForSequenceClassification
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from transformers import BertTokenizer, GPT2LMHeadModel
from datasets import load_dataset, load_metric
from transformers import TrainingArguments
from transformers import Trainer
import numpy as np
import evaluate
import random

checkpoint = "gpt2-xl"
configuration = GPT2Config()
tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
model = GPT2ForSequenceClassification(configuration).from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id


raw_datasets = load_dataset("glue", "cola")

def tokenize_function(example):
    #return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
    return tokenizer(example["sentence"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments("test-trainer")


trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_datasets["train"],
    eval_dataset = tokenized_datasets["validation"],
    data_collator = data_collator,
    tokenizer = tokenizer
)

trainer.train()

metric = load_metric("glue", "cola")


def get_preds():
    predictions = trainer.predict(tokenized_datasets["validation"])
    preds = np.argmax(preds.predictions, axis=-1)
    return metric.compute(predictions = preds, references = predictions.label_ids)

def flatten_model(modules):
    def flatten_list(_2d_list):
        flat_list = []
        # Iterate through the outer list
        for element in _2d_list:
            if type(element) is list:
                # If the element is of type list, iterate through the sublist
                for item in element:
                    flat_list.append(item)
            else:
                flat_list.append(element)
        return flat_list

    ret = []
    try:
        for _, n in modules:
            ret.append(loopthrough(n))
    except:
        try:
            if str(modules._modules.items()) == "odict_items([])":
                ret.append(modules)
            else:
                for _, n in modules._modules.items():
                    ret.append(loopthrough(n))
        except:
            ret.append(modules)
    return flatten_list(ret)



def do_prune():
    results_list = []
    module_list = [module for module in model.modules() ]
    target_layers =[]
    # this is needed
    flatted_list= flatten_model(module_list)
    for count, value in enumerate(flatted_list):
        if isinstance(value, (nn.Conv2d,nn.AvgPool2d,nn.BatchNorm2d, nn.Linear, nn.MultiheadAttention)):
            target_layers.append((value, 'weight'))
    random.shuffle(target_layers)
    chunks = [target_layers[x:x+10] for x in range(0, len(target_layers), 10)]
    for sparsity in [0,0.1, 0.4555555555, 0.8,0.5, 0.8]:
        for chunk in chunks:
            prune.global_unstructured(
                chunk,
                pruning_method = prune.L1Unstructured,
                amount = sparsity
            )
        results_list.append(get_preds())
    return results_list


pruned_performance = pd.DataFrame(do_prune())
pruned_performance.to_csv("gpt2_cola_results.csv", index=False)



