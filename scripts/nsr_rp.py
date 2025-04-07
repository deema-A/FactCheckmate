import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import torch
from torch.utils.data import DataLoader, Dataset
import argparse
import os


MODELS_IDS = {
  "llama2_7b": "meta-llama/Llama-2-7b-hf",
  "llama2_13b": "meta-llama/Llama-2-13b-hf",
  "mistral_7b": "mistralai/Mistral-7B-v0.3",
  "llama3_8b": "meta-llama/Meta-Llama-3-8B",
  "llama3_70b": "meta-llama/Meta-Llama-3-70B",
  "gemma_7b": "google/gemma-7b",
  "llama3.1_8b": "meta-llama/Meta-Llama-3.1-8B"
}

def collate_fn(batch):
    data, labels= zip(*batch)
    data_tensor = torch.stack(data)
    labels_tensor = torch.stack(labels)
    return data_tensor, labels_tensor

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label
    
def map_selected_mode(hs, mode):
    # Convert list of tensors into a single tensor
    # Each tensor is of shape [1, 4096], we concatenate along dim=0
    tensor = torch.cat(hs, dim=0).float() 

    # Squeeze the middle dimension to make the shape [len(hs), 4096]
    tensor = tensor.squeeze(1)  # Now tensor shape is [len(hs), 4096]

    if mode == 0:
        # Mean pooling across the new concatenated dimension
        mean_pooled = torch.mean(tensor, dim=0)
        assert mean_pooled.shape[-1] == 4096
        return mean_pooled
    
    elif mode == 1:
        # Max pooling across the new concatenated dimension
        max_pooled, _ = torch.max(tensor, dim=0)
        assert max_pooled.shape[-1] == 4096
        return max_pooled
    
    else:
        # Return the last sequence element of the last tensor if mode is neither 0 nor 1
        last = hs[-1].squeeze(0)
        assert last.shape[-1] == 4096
        return last  # Ensure it returns a tensor of shape [4096]
    


parser = argparse.ArgumentParser(description="Process layer number and tr value.")

# Add arguments
parser.add_argument('-l', '--layer_number', type=int, required=True, help="Layer number (must be a non-negative integer).")
parser.add_argument('-t', '--tr', type=str, required=True, help="tr value (must be between 0 and 1).")
parser.add_argument('-m', '--mode', type=int, default=0, help="mode")
parser.add_argument('-mi', '--model_id', type=str, default="llama2_7b", help="mode")
parser.add_argument('-d', "--dataset_name", type=str, default="nq_open", help="Enter the dataset name")


# Parse arguments
args = parser.parse_args()
layer = args.layer_number
tr = float(args.tr)
mode = args.mode
model_id = args.model_id
dataset_name = args.dataset_name

train_path = f"datasets/nq_open/Llama-2-7b-hf/hs/train_fct/layer_{layer}.pth"
val_path = f"datasets/nq_open/Llama-2-7b-hf/hs/val_fct/layer_{layer}.pth"

train_data1 = torch.load(train_path)
val_data1 = torch.load(val_path)

train_data = [(map_selected_mode(tensors, mode), map_selected_mode(target, mode)) for tensors, target in zip(train_data1['hs'], train_data1['hs_fct'])]
val_data = [(map_selected_mode(tensors, mode), map_selected_mode(target, mode)) for tensors, target in zip(val_data1['hs'], val_data1['hs_fct'])]

# Creating datasets using the correct format
train_dataset = CustomDataset([t[0] for t in train_data], [t[1] for t in train_data])
val_dataset = CustomDataset([t[0] for t in val_data], [t[1] for t in val_data])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)

############################################################################################################
model_id = MODELS_IDS[args.model_id].split("/")[-1] # update the model id to be part of the path

# Check if the directory exists, and create it if it doesn't
os.makedirs(f"nsrs/{dataset_name}/{model_id}", exist_ok=True)

############################################################################################################



run_name = f"l_{layer}_m_{mode}_b_128_rp_tr_{tr}"
wandb.init(project=f"fixed_adjstmnts", entity="deema2", name=run_name, reinit=True)

best_model_path = f"nsrs/{dataset_name}/{model_id}/{run_name}.pth"


# Model definition
class ReparameterizedModel(nn.Module):
    def __init__(self):
        super(ReparameterizedModel, self).__init__()
        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.mean = nn.Linear(512, 4096)
        self.log_var = nn.Linear(512, 4096)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, data):
        x = torch.relu(self.fc1(data))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        mean = self.mean(x)
        log_var = self.log_var(x)
        noise = self.reparameterize(mean, log_var)
        return noise

# Initialize model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
model = ReparameterizedModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

best_val_loss = float('inf')
epochs = 100

for epoch in range(epochs):
    print(f"train epoch: {epoch}")
    model.train()
    train_loss = 0
    count = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs.float())
        inputs2 = inputs+  (tr * outputs)
        
        loss = criterion(inputs2, targets)
        
        # loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        count += 1

    train_loss /= count
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")

    # Validation phase
    model.eval()
    val_loss = 0
    count = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs.float())
            
            inputs2 = inputs +  (tr * outputs)
            
            loss = criterion(inputs2, targets)
            
            val_loss += loss.item()
            count += 1

    val_loss /= count
    print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}")

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print("Saved best model")