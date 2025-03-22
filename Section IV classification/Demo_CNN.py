import torch
from torch import nn, optim  
from torch.utils.data import Dataset, DataLoader, Subset 
from sklearn.model_selection import StratifiedKFold  
from sklearn.preprocessing import StandardScaler  
import scipy.io
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import numpy as np
mat_file_path = 'He_vs_dm_stft.mat'

class CustomDataset(Dataset):
    def __init__(self, data, targets):
   
        self.data = torch.tensor(data, dtype=torch.float32) 
        self.targets = torch.tensor(targets, dtype=torch.long)  

    def __len__(self):
  
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()  
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1) 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  
        self.fc1 = nn.Linear(32 * 7 * 5, 128)  
        self.fc2 = nn.Linear(128, 2)  

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  
        x = x.view(-1, 32 * 7 * 5)  
        x = torch.relu(self.fc1(x))  
        x = self.fc2(x) 
        return x

X = scipy.io.loadmat(mat_file_path)
X = X["data"]
zeros_part = np.zeros(200)
ones_part = np.ones(200)
y = np.concatenate((zeros_part, ones_part))

X = X.reshape(-1, 1, 15, 10)  
X = StandardScaler().fit_transform(X.reshape(-1, 150)).reshape(-1, 1, 15, 10) 
dataset = CustomDataset(X, y)  

fold = 0
all_metrics = {'accuracy': [], 'g_mean': [], 'f1_score': [], 'precision': []}
for k in range(5):
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X.reshape(-1, 150), y)):  
        print(f"Fold {fold + 1}")  
        train_subsampler = Subset(dataset, train_idx)  
        val_subsampler = Subset(dataset, val_idx) 

        train_loader = DataLoader(train_subsampler, batch_size=16, shuffle=True) 
        val_loader = DataLoader(val_subsampler, batch_size=16, shuffle=False)  

        model = SimpleCNN()  
        criterion = nn.CrossEntropyLoss() 
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(10):  
            model.train()  
            for i, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()  
                outputs = model(inputs) 
                loss = criterion(outputs, labels) 
                loss.backward()  
                optimizer.step()  
            model.eval() 
            all_labels = []
            all_preds = []
            correct = 0
            total = 0
            with torch.no_grad(): 
                for inputs, labels in val_loader:
                    outputs = model(inputs)  
                    _, predicted = torch.max(outputs.data, 1) 
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(predicted.cpu().numpy())

            acc = accuracy_score(all_labels, all_preds)
            cm = confusion_matrix(all_labels, all_preds)
            tn, fp, fn, tp = cm.ravel()
            g_mean = np.sqrt((tp / (tp + fn)) * (tn / (tn + fp)))
            f1 = f1_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds)

            all_metrics['accuracy'].append(acc)
            all_metrics['g_mean'].append(g_mean)
            all_metrics['f1_score'].append(f1)
            all_metrics['precision'].append(precision)

average_accuracy = np.mean(all_metrics['accuracy'])
std_accuracy = np.std(all_metrics['accuracy'])

average_g_mean = np.mean(all_metrics['g_mean'])
std_g_mean = np.std(all_metrics['g_mean'])

average_f1_score = np.mean(all_metrics['f1_score'])
std_f1_score = np.std(all_metrics['f1_score'])

average_precision = np.mean(all_metrics['precision'])
std_precision = np.std(all_metrics['precision'])


print(
    '& {:.2f}'.format(average_precision * 100) + '$_{\pm' + '{:.2f}'.format(std_precision * 100) + '}$' +
    ' & {:.2f}'.format(average_g_mean * 100) +   '$_{\pm' + '{:.2f}'.format(std_g_mean * 100)    + '}$' +
    ' & {:.2f}'.format(average_f1_score * 100) + '$_{\pm' + '{:.2f}'.format(std_f1_score * 100)  + '}$' +
    ' & {:.2f}'.format(average_accuracy * 100) + '$_{\pm' + '{:.2f}'.format(std_g_mean * 100)    + '}$'
)
