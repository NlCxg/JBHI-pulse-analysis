import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score
import numpy as np
import scipy.io
mat_file_path = 'He_vs_dm_stft.mat'

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 位置编码类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

##### start  加载数据集 #######
XX = scipy.io.loadmat(mat_file_path)
data = XX["data"]
sample_num = data.shape[0]
feature_dim = data.shape[1]
input_dim = data.shape[1]
print(f"Dimen is {input_dim}, Sample_size is {sample_num} ")
#print(f"Average 5 - fold Cross - Validation Accuracy: {average_accuracy}%")
zeros_part = np.zeros(200)
ones_part = np.ones(200)
labels = np.concatenate((zeros_part, ones_part))

# 超参数设置
d_model = 150
nhead = 2
num_layers = 1
num_classes = 2
batch_size = 16
epochs = 10
learning_rate = 0.001

# Transformer 分类器类
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, num_classes):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        #x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

# 计算 G-mean
def gmean(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return np.sqrt(sensitivity * specificity)

# 五折交叉验证
all_accuracies = []
all_f1_scores = []
all_gmeans = []
all_precisions = []
all_metrics = {'accuracy': [], 'g_mean': [], 'f1_score': [], 'precision': []}

for k in range(20):
    fold = 0
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    for train_index, test_index in skf.split(data, labels):
        fold += 1
        print(f"Fold {fold}")

        # 划分训练集和测试集
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # 创建数据集和数据加载器
        train_dataset = CustomDataset(X_train, y_train)
        test_dataset = CustomDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 初始化模型、损失函数和优化器
        model = TransformerClassifier(input_dim, d_model, nhead, num_layers, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 训练模型
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs.unsqueeze(1))
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            #print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")

        # 评估模型
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs.unsqueeze(1))
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(targets.numpy())
                y_pred.extend(predicted.numpy())

        # 计算评价指标
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        g_mean = gmean(y_true, y_pred)
        # print(accuracy)
        precision = precision_score(y_true, y_pred)

        all_metrics['accuracy'].append(accuracy)
        all_metrics['g_mean'].append(g_mean)
        all_metrics['f1_score'].append(f1)
        all_metrics['precision'].append(precision)

# 计算各指标的平均值
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
    ' & {:.2f}'.format(average_accuracy * 100) + '$_{\pm' + '{:.2f}'.format(std_accuracy * 100)    + '}$'
)
