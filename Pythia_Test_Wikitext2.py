# Import libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from tqdm.auto import tqdm
from torch.cuda.amp import autocast  # For mixed precision

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load the Wikitext-2 dataset (test split)
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

# Load the tokenizer and model
model_name = 'EleutherAI/pythia-1.4b'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Set pad_token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize the texts
def tokenize_function(examples):
    return tokenizer(examples['text'])

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

# Group texts into chunks of block_size
block_size = 512

def group_texts(examples):
    # Concatenate all input_ids
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples['input_ids'])
    # Drop the small remainder
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result

lm_dataset = tokenized_dataset.map(group_texts, batched=True)

# Define the DataCollator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Create the DataLoader
batch_size = 8  # Adjust this value based on your GPU memory
data_loader = DataLoader(lm_dataset, batch_size=batch_size, collate_fn=data_collator)

# Compute perplexity with mixed precision
model.eval()
total_loss = 0
total_tokens = 0

with torch.no_grad():
    for batch in tqdm(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast():  # Enable mixed precision
            outputs = model(**batch)
            loss = outputs.loss
            num_tokens = (batch['labels'] != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
print(f'Perplexity: {perplexity}')
