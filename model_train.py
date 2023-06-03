#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, average_precision_score


# In[2]:


# Define GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)
        return out
    def predict(self, input_data):
        input_data = torch.tensor(input_data).float().to(device)
        output = self.forward(input_data)
        _, predicted = torch.max(output.data, 1)
        return predicted.item()


# In[3]:


# Load dataset
df = pd.read_csv("updated_data_final.csv")


# In[4]:


df.loc[df["status"] == "phishing", "status"] = 1
df.loc[df["status"] == "legitimate", "status"] = 0


# In[5]:


df.replace([np.inf, -np.inf], np.nan, inplace=True)


# In[6]:


#Remove duplicates
df = df.drop_duplicates()


# In[7]:


df.status.value_counts()


# In[8]:


# Define character dictionary
alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\t\n\x0b\x0c"
char_dict = {}
for i, char in enumerate(alphabet):
    char_dict[char] = i


# In[9]:


# Define custom dataset
class PhishingDataset(Dataset):
    def __init__(self, df, char_dict):
        self.df = df
        self.char_dict = char_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        url = self.df.iloc[index]["url"]
        label = self.df.iloc[index]["status"]
        feature = torch.zeros(200, 100)
        for i, char in enumerate(url):
            if i >= 200:
                break
            if char in self.char_dict:
                feature[i, self.char_dict[char]] = 1
            
        return feature, label


# In[10]:


# Split dataset into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["status"], random_state=42)


# In[11]:


# Create custom datasets and dataloaders
train_dataset = PhishingDataset(train_df, char_dict)
test_dataset = PhishingDataset(test_df, char_dict)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# In[12]:


# Define hyperparameters
input_size = 100
hidden_size = 64
output_size = 2
num_layers = 2
batch_size = 32
epoch = 10
learning_rate = 0.001


# In[13]:


# Initialize GRU model
model = GRUModel(input_size, hidden_size, output_size, num_layers)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# In[14]:


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# In[15]:


# Training loop

for e in range(epoch):
    running_loss = 0.0
    model.train()
    for features, labels in train_loader:
        features = features.float()
        labels = torch.tensor(labels).long() 
        features =  features.to(device)
        labels = labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item()

    # Calculate and print average training loss for current epoch
    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch [{e+1}/{epoch}], Loss: {avg_train_loss:.4f}")


# In[16]:


# Evaluation loop
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for features, labels in test_loader:
        features = features.float()
        labels = torch.tensor(labels).long() # Convert tuple to torch tensor
        features = features.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)

        # Collect true and predicted labels for calculating evaluation measures
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())


# In[17]:


# Calculate evaluation measures
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
mean_avg_precision = average_precision_score(y_true, y_pred)


# In[18]:


# Print evaluation measures
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Mean Average Precision: {mean_avg_precision:.4f}")


# In[ ]:





# In[23]:


import numpy as np
# Convert URL to feature matrix
url = "https://www.google.co.in/"
feature = torch.zeros(1, 200, 100)
for i, char in enumerate(url):
    if i >= 200:
        break
    if char in char_dict:
        feature[0, i, char_dict[char]] = 1
    else:
        feature[0, i, char_dict[' ']] = 1

# Load feature matrix into PyTorch tensor and move to device
feature = feature.float().to(device)

# Pass tensor to trained model and get predicted class probabilities
model.eval()
with torch.no_grad():
    output = model(feature)
probabilities = output.cpu().numpy()[0]

# Classify URL as phishing if predicted class is phishing and probability is above a threshold
threshold = 0.5
if probabilities[1] > threshold:
    print(f"The URL {url} is classified as phishing.")
else:
    print(f"The URL {url} is classified as Legitimate.")


# In[24]:


model_path = 'model.pth'


# In[25]:


# Save the entire model
torch.save(model, model_path)


# In[26]:


# Save the trained model
torch.save(model.state_dict(), 'model.pth')


# In[ ]:




