import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score  
import torch.optim as optim
import wandb
import argparse
import os

# Set a seed value
seed = 42  # You can choose any seed number

# Set the random seed for various libraries
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if(torch.cuda.is_available()):
    gpu=0
    device='cuda:{}'.format(gpu)
else:
  device='cpu' 


# MODES = [0, -1, 1]
MODEL_LAYER_EMB_MAP = {
    "llama2_7b": {
        "num_layers": 32,
        "emb_size": 4096
    },
    "llama2_7b_chat": {
        "num_layers": 32,
        "emb_size": 4096
    },
    "llama2_13b": {
        "num_layers": 40,
        "emb_size": 5120
    },
    "llama2_13b_chat": {
        "num_layers": 40,
        "emb_size": 5120
    },
    "mistral_7b": {
        "num_layers": 32,
        "emb_size": 4096
    },
    "mistral_7b_instruct": {
        "num_layers": 32,
        "emb_size": 4096
    },
    "llama3_8b": {
        "num_layers": 32,
        "emb_size": 4096
    },
    "llama3_8b_instruct": {
        "num_layers": 32,
        "emb_size": 4096
    },
    "llama3_70b": {
        "num_layers": 80,
        "emb_size": 8192
    },
    "llama3_70b_instruct": {
        "num_layers": 80,
        "emb_size": 8192
    },
    "gemma_7b": {
        "num_layers": 28,
        "emb_size": 3072
    },
    "gemma_7b_it": {
        "num_layers": 28,
        "emb_size": 3072
    },
    "llama3.1_8b": {
        "num_layers": 32,
        "emb_size": 4096
    },
    "llama3.1_8b_instruct": {
        "num_layers": 32,
        "emb_size": 4096
    }
}

MODELS_IDS = {
  "llama2_7b": "meta-llama/Llama-2-7b-hf",
  "llama2_13b": "meta-llama/Llama-2-13b-hf",
  "mistral_7b": "mistralai/Mistral-7B-v0.3",
  "llama3_8b": "meta-llama/Meta-Llama-3-8B",
  "llama3_70b": "meta-llama/Meta-Llama-3-70B",
  "gemma_7b": "google/gemma-7b",
  "llama3.1_8b": "meta-llama/Meta-Llama-3.1-8B",
  "llama2_7b_chat":"meta-llama/Llama-2-7b-chat-hf",
  "llama2_13b_chat": "meta-llama/Llama-2-13b-chat-hf",
  "mistral_7b_instruct":"mistralai/Mistral-7B-Instruct-v0.3",
  "llama3_8b_instruct":"meta-llama/Meta-Llama-3-8B-Instruct",
  "llama3_70b_instruct": "meta-llama/Meta-Llama-3-70B-Instruct",
  "gemma_7b_it": "google/gemma-7b-it",
  "llama3.1_8b_instruct": "meta-llama/Llama-3.1-8B-Instruct"
}

def map_selected_mode(hs, mode):
    # Convert list of tensors into a single tensor
    # Each tensor is of shape [1, 4096], we concatenate along dim=0
    tensor = torch.cat(hs, dim=0).float()  # Resulting tensor shape will be [len(hs), 1, 4096]

    # Squeeze the middle dimension to make the shape [len(hs), 4096]
    tensor = tensor.squeeze(1)  # Now tensor shape is [len(hs), 4096]

    if mode == 0:
        # Mean pooling across the new concatenated dimension
        mean_pooled = torch.mean(tensor, dim=0)
        # print("mean_pooled.shape[-1]", mean_pooled.shape[-1])
        assert mean_pooled.shape[-1] == emb_size #5120
        return mean_pooled
    
    elif mode == 1:
        # Max pooling across the new concatenated dimension
        max_pooled, _ = torch.max(tensor, dim=0)
        # print("max_pooled.shape[-1]", max_pooled.shape[-1])
        assert max_pooled.shape[-1] == emb_size #5120
        return max_pooled
    
    else:
        # Return the last sequence element of the last tensor if mode is neither 0 nor 1
        last = hs[-1].squeeze(0)
        assert last.shape[-1] == emb_size #5120
        return last  # Ensure it returns a tensor of shape [4096]
    

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.relu = nn.ReLU()
        self.fc = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_size, output_size)  # Hidden to Output Layer
        self.sigmoid = nn.Sigmoid()  # Since it's a binary classification

    def forward(self, x):
        out = self.relu(self.fc(x))
        out = self.dropout(out)
        out = self.sigmoid(self.fc2(out))
        return out
    
# (tensor_id, tensor, label)
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
    
def collate_fn(batch):
    data, labels = zip(*batch)
    if all(isinstance(d, torch.Tensor) for d in data):
        data_tensor = torch.stack(data)
    else:
        data_tensor = torch.tensor(data)
    labels_tensor = torch.tensor(labels)
    return data_tensor, labels_tensor

def accuracy(outputs, labels):
    rounded_preds = torch.round(outputs)
    correct = (rounded_preds == labels).float()  # Convert into float for division
    acc = correct.sum() / len(correct)
    return acc

def f1(outputs, labels):
    predictions = torch.round(outputs.detach())
    f1_score_value = f1_score(labels.cpu().numpy(), predictions.cpu().numpy(), average='macro', zero_division=0)
    return f1_score_value

def precision(outputs, labels):
    rounded_preds = torch.round(outputs)
    true_positives = ((rounded_preds == 1) & (labels == 1)).float().sum()
    predicted_positives = (rounded_preds == 1).float().sum()
    precision = true_positives / predicted_positives if predicted_positives != 0 else 0
    return precision

def recall(outputs, labels):
    rounded_preds = torch.round(outputs)
    true_positives = ((rounded_preds == 1) & (labels == 1)).float().sum()
    actual_positives = (labels == 1).float().sum()
    recall = true_positives / actual_positives if actual_positives != 0 else 0
    return recall

def false_positives(outputs, labels):
    rounded_preds = torch.round(outputs)
    false_positives = ((rounded_preds == 1) & (labels == 0)).float().sum()
    return false_positives.item()

def false_negatives(outputs, labels):
    rounded_preds = torch.round(outputs)
    false_negatives = ((rounded_preds == 0) & (labels == 1)).float().sum()
    return false_negatives.item()


def train_model(model, train_loader, val_loader, epochs, criterion, optimizer, best_model_path):
    
    best_val_accuracy = 0.0
    best_val_loss = 1000
    model.to(device)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss, train_acc, train_f1, train_precision, train_recall, train_false_positives, train_false_negatives = 0, 0, 0, 0, 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.float())
            loss = criterion(outputs.squeeze(-1), labels.float())

            train_loss += loss.item()
            train_acc += accuracy(outputs.squeeze(-1), labels.float())
            train_f1 += f1(outputs.squeeze(-1), labels.float())

            train_precision += precision(outputs.squeeze(-1), labels.float())
            train_recall += recall(outputs.squeeze(-1), labels.float())
            train_false_positives += false_positives(outputs.squeeze(-1), labels.float())
            train_false_negatives += false_negatives(outputs.squeeze(-1), labels.float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_f1 /= len(train_loader)

        train_precision/= len(train_loader)
        train_recall/= len(train_loader)
        train_false_positives/= len(train_loader)
        train_false_negatives/= len(train_loader)
        
        # print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Train F1: {train_f1}')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Train F1: {train_f1:.4f}, '
            f'Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, '
            f'Train False Positives: {train_false_positives:.4f}, Train False Negatives: {train_false_negatives:.4f}')

        # Validation phase
        model.eval()
        val_loss, val_acc, val_f1, val_precision, val_recall, val_false_positives, val_false_negatives = 0, 0, 0, 0, 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs.float())
                loss = criterion(outputs.squeeze(-1), labels.float())
                val_loss += loss.item()
                val_acc += accuracy(outputs.squeeze(-1), labels.float())
                val_f1 += f1(outputs.squeeze(-1), labels.float())

                val_precision += precision(outputs.squeeze(-1), labels.float())
                val_recall += recall(outputs.squeeze(-1), labels.float())
                val_false_positives += false_positives(outputs.squeeze(-1), labels.float())
                val_false_negatives += false_negatives(outputs.squeeze(-1), labels.float())
                

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        val_f1 /= len(val_loader)

        val_precision/= len(train_loader)
        val_recall/= len(train_loader)
        val_false_positives/= len(train_loader)
        val_false_negatives/= len(train_loader)

        # print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Val F1: {val_f1}')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Val F1: {val_f1:.4f}, '
            f'Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, '
            f'Val False Positives: {val_false_positives:.4f}, Val False Negatives: {val_false_negatives:.4f}')


        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'train_f1': train_f1,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_false_positives': train_false_positives,
            'train_false_negatives': train_false_negatives,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_f1': val_f1,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_false_positives': val_false_positives,
            'val_false_negatives': val_false_negatives
        })

        # if val_loss < best_val_loss:
        if val_acc > best_val_accuracy:
            # best_val_loss = val_loss
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"** New best model saved at Epoch {epoch} with Val Accuracy: {val_acc} and Val Loss: {val_loss} **")

    return


def test_model(model, test_loader, criterion):
    model.to(device)
    model.eval()
    test_loss, test_acc, test_f1, test_precision, test_recall, test_false_positives, test_false_negatives = 0, 0, 0, 0, 0, 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.float())

            test_loss += criterion(outputs.squeeze(-1), labels.float()).item()
            test_acc += accuracy(outputs.squeeze(-1), labels.float())
            test_f1 += f1(outputs.squeeze(-1), labels.float())
            
            test_precision += precision(outputs.squeeze(-1), labels.float())
            test_recall += recall(outputs.squeeze(-1), labels.float())
            test_false_positives += false_positives(outputs.squeeze(-1), labels.float())
            test_false_negatives += false_negatives(outputs.squeeze(-1), labels.float())


    test_loss /= len(test_loader)
    test_acc /= len(test_loader)
    test_f1 /= len(test_loader)

    test_precision /= len(test_loader)
    test_recall /= len(test_loader)
    test_false_positives /= len(test_loader)
    test_false_negatives /= len(test_loader)

    test_metrics = {
    'test_loss': test_loss,
    'test_acc': test_acc,
    'test_f1': test_f1,
    'test_precision': test_precision,
    'test_recall': test_recall,
    'test_false_positives': test_false_positives,
    'test_false_negatives': test_false_negatives
}
    wandb.log(test_metrics)

    # print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test F1: {test_f1}')
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}, '
      f'Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, '
      f'Test False Positives: {test_false_positives:.4f}, Test False Negatives: {test_false_negatives:.4f}')


parser = argparse.ArgumentParser(description="model_id and dataset name.")
parser.add_argument("model_id", type=str, help="Enter the model ID")
parser.add_argument("dataset_name", type=str, help="Enter the dataset name")
parser.add_argument("modes", nargs='*', type=int, default=[0],
                        help="Pass the modes, e.g., 0, 1, or -1. Accepts multiple modes separated by space.")
# [0, -1, 1] 

#modes [0, -1, 1]
# Parse arguments
args = parser.parse_args()
model_id = args.model_id
dataset_name = args.dataset_name
modes = args.modes

emb_size = MODEL_LAYER_EMB_MAP[model_id]["emb_size"]
num_layers = MODEL_LAYER_EMB_MAP[model_id]["num_layers"]
model_id = MODELS_IDS[args.model_id].split("/")[-1] # update the model id to be part of the path

layers = list(range(num_layers+1))

# Check if the directory exists, and create it if it doesn't
os.makedirs(f"clss/{dataset_name}/{model_id}", exist_ok=True)

for mode in modes:
    for layer in layers:

        # train_path = f"datasets/{dataset_name}/{model_id}/hs/train/layer_{layer}.pth"
        # val_path = f"datasets/{dataset_name}/{model_id}/hs/val/layer_{layer}.pth"
        # test_path = f"datasets/{dataset_name}/{model_id}/hs/test/layer_{layer}.pth"
        
        train_path = f"/projects/bbwz/deema/additional_LMs/{model_id}/hs/train/layer_{layer}.pth"
        val_path = f"/projects/bbwz/deema/additional_LMs/{model_id}/hs/val/layer_{layer}.pth"
        test_path = f"/projects/bbwz/deema/additional_LMs/{model_id}/hs/test/layer_{layer}.pth"
        
        # train_path = f"/projects/bbwz/deema/additional_LMs/train_factual_aggregated/layer_{layer}.pth"
        # val_path = f"/projects/bbwz/deema/additional_LMs/val_factual_aggregated/layer_{layer}.pth"
        # test_path = f"/projects/bbwz/deema/additional_LMs/test_factual_aggregated/layer_{layer}.pth"

        train_data = torch.load(train_path, map_location=torch.device('cpu'))
        val_data = torch.load(val_path, map_location=torch.device('cpu'))
        test_data = torch.load(test_path, map_location=torch.device('cpu'))

        train_data = [(map_selected_mode(tensors, mode), label) for tensors, label in zip(train_data['hs'], train_data['labels'])]
        val_data = [(map_selected_mode(tensors, mode), label) for tensors, label in zip(val_data['hs'], val_data['labels'])]
        test_data = [(map_selected_mode(tensors, mode), label) for tensors, label in zip(test_data['hs'], test_data['labels'])]

        print("len train_data: ", len(train_data))
        print("len val_data: ", len(val_data))
        print("len test_data: ", len(test_data))

        # Creating datasets using the correct format
        train_dataset = CustomDataset([t[0] for t in train_data], [t[1] for t in train_data])
        val_dataset = CustomDataset([t[0] for t in val_data], [t[1] for t in val_data])
        test_dataset = CustomDataset([t[0] for t in test_data], [t[1] for t in test_data])

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)

        run_name = f"m_{mode}_l_{layer}_b_128"
        wandb.init(project=f"cls_{dataset_name}_{model_id}_3_shots", entity="deema2", name=run_name, reinit=True)

        best_model_path = f"clss/{dataset_name}/{model_id}/{run_name}.pth"

        model = SimpleMLP(emb_size, 1024, 1)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        train_model(model, train_loader, val_loader, epochs=50, criterion=criterion, optimizer=optimizer, best_model_path = best_model_path)
        model.load_state_dict(torch.load(best_model_path))
        test_model(model, test_loader, criterion)