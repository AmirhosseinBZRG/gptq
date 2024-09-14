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

# Load the HellaSwag dataset (validation split)
dataset = load_dataset('hellaswag', split='validation')

# Load the tokenizer and model
model_name = 'EleutherAI/pythia-1.4b'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Set pad_token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Preprocess function to concatenate context and correct ending
def preprocess_function(examples):
    texts = [ctx + " " + endings[int(label)] for ctx, endings, label in zip(examples['ctx'], examples['endings'], examples['label'])]
    return {'text': texts}

# Apply the preprocess function to the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

# Tokenize the texts
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=512)

tokenized_dataset = tokenized_dataset.map(tokenize_function, batched=True, remove_columns=['text'])

# Define the DataCollator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Create the DataLoader
batch_size = 8  # Adjust this value based on your GPU memory
data_loader = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=data_collator)

# Compute perplexity
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
