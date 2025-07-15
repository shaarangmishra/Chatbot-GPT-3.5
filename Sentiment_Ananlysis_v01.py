import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
print("All libraries imported successfully!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Standfor Sentiment Treebank (SST-2) dataset is used here. You can download it from https://gluebenchmark.com/tasks. Train, validation and test files
# are stored as tab-separated files
# The data consists of movies reviews with corresponding binary labels (0: negative and 1: positive)
train_data = pd.read_csv(" ", sep='\t')
val_data = pd.read_csv(" ", sep='\t')

# Define a custom dataset class
class MovieReviewDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_length):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = str(self.reviews[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        dict_enc = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
        return dict_enc

# Initialize BERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.to(device)

# Define max sequence length and batch size
MAX_LENGTH = 256
BATCH_SIZE = 32

# Create datasets and dataloaders
train_dataset = MovieReviewDataset(
    reviews=train_data['sentence'],
    labels=train_data['label'],
    tokenizer=tokenizer,
    max_length=MAX_LENGTH
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

val_dataset = MovieReviewDataset(
    reviews=val_data['sentence'],
    labels=val_data['label'],
    tokenizer=tokenizer,
    max_length=MAX_LENGTH
)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# Training loop
def train_loop (train_loader, model, optimizer, loss_fn, train_loss, scaler, device, epoch, NUM_EPOCHS):
  model.train()
  progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{NUM_EPOCHS}', leave=False)
  sum_loss = 0
  for batch in progress_bar:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['label'].to(device)

    #forward
    with torch.cuda.amp.autocast():
      outputs = model(input_ids, attention_mask=attention_mask)
      logits = outputs.logits
      loss = loss_fn(logits, labels)
      sum_loss += loss.item()

    #backward
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

  train_loss.append(sum_loss/len(train_loader))
  progress_bar.set_postfix({'Avg. training_loss': (sum_loss/len(train_loader))})

  return (train_loss)


# Validation loop
def val_loop (val_loader, model, device, best, val_arr, loss_fn, filename, epoch, NUM_EPOCHS):
  model.eval()
  total_correct = 0
  total_samples = 0
  val_loss = 0
  progress_bar = tqdm(val_loader, desc=f'Validation Epoch {epoch+1}/{NUM_EPOCHS}', leave=False)
  with torch.no_grad():
    for batch in val_loader:
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      labels = batch['label'].to(device)

      outputs = model(input_ids, attention_mask=attention_mask)
      _, predicted = torch.max(outputs.logits, 1)
      logits = outputs.logits
      loss1 = loss_fn(logits, labels)

      val_loss += loss1.item()
      total_correct += (predicted == labels).sum().item()
      total_samples += labels.size(0)

  progress_bar.set_postfix({'Avg. validation loss': (val_loss/len(val_loader))})

  accuracy = total_correct / total_samples
  print(f'Validation Accuracy: {accuracy:.4f}')
  val_arr.append((val_loss/len(val_loader)))

  # Save only the best perfoming model weights
  if (val_loss/len(val_loader))<=best:
    state = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
    torch.save(state, filename)
    best = val_loss/len(val_loader)

  return (best, val_arr)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()

# Training loop
NUM_EPOCHS = 5

train_loss = []
val_arr = []
best = 1
filename = ' ' # To store trained model weights
for epoch in range(NUM_EPOCHS):
  train_loss = train_loop (train_loader, model, optimizer, loss_fn, train_loss, scaler, device, epoch, NUM_EPOCHS)
  best, val_arr = val_loop (val_loader, model, device, best, val_arr, loss_fn, filename, epoch, NUM_EPOCHS)

epc = []
i = 0
for i in range (NUM_EPOCHS):
  epc.append(i)

plt.figure(figsize=(5,5))
plt.title('Loss v/s Epoch')
plt.plot(NUM_EPOCHS, train_loss, label='Train')
plt.plot(NUM_EPOCHS, val_arr, label='val')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()