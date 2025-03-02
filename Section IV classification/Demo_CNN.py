import torch
from torch import nn, optim  # 神经网络模块和优化器模块
from torch.utils.data import Dataset, DataLoader, Subset  # 数据集和数据加载器
from sklearn.model_selection import StratifiedKFold  # 用于分层K折交叉验证
from sklearn.preprocessing import StandardScaler  # 数据标准化
import scipy.io
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import numpy as np
mat_file_path = 'He_vs_dm_stft.mat'
# 自定义数据集类，继承自Dataset
class CustomDataset(Dataset):
    def __init__(self, data, targets):
        # 初始化函数，接收数据和标签
        self.data = torch.tensor(data, dtype=torch.float32)  # 将数据转换为tensor并设置类型
        self.targets = torch.tensor(targets, dtype=torch.long)  # 将标签转换为long类型的tensor

    def __len__(self):
        # 返回数据集大小
        return len(self.data)

    def __getitem__(self, idx):
        # 根据索引获取单个样本及其标签
        return self.data[idx], self.targets[idx]


# 定义简单的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()  # 继承nn.Module的构造函数
        # 假设输入reshape为1x24x16的图像格式
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 第一层卷积层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 最大池化层
        self.fc1 = nn.Linear(32 * 7 * 5, 128)  # 全连接层，根据输入尺寸调整
        self.fc2 = nn.Linear(128, 2)  # 输出层为2个类别，对应二分类

    def forward(self, x):
        # 前向传播过程
        x = self.pool(torch.relu(self.conv1(x)))  # 应用激活函数后进行池化
        x = x.view(-1, 32 * 7 * 5)  # 展开特征图
        x = torch.relu(self.fc1(x))  # 全连接层后应用ReLU激活函数
        x = self.fc2(x)  # 输出层
        return x

X = scipy.io.loadmat(mat_file_path)
X = X["data"]
zeros_part = np.zeros(200)
ones_part = np.ones(200)
y = np.concatenate((zeros_part, ones_part))

X = X.reshape(-1, 1, 15, 10)  # 调整形状以适应CNN输入
X = StandardScaler().fit_transform(X.reshape(-1, 150)).reshape(-1, 1, 15, 10)  # 数据标准化
dataset = CustomDataset(X, y)  # 创建自定义数据集实例

fold = 0
all_metrics = {'accuracy': [], 'g_mean': [], 'f1_score': [], 'precision': []}
for k in range(5):
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X.reshape(-1, 150), y)):  # 进行分割
        print(f"Fold {fold + 1}")  # 打印当前是第几折
        train_subsampler = Subset(dataset, train_idx)  # 获取训练子集
        val_subsampler = Subset(dataset, val_idx)  # 获取验证子集

        train_loader = DataLoader(train_subsampler, batch_size=16, shuffle=True)  # 创建训练数据加载器
        val_loader = DataLoader(val_subsampler, batch_size=16, shuffle=False)  # 创建验证数据加载器

        model = SimpleCNN()  # 实例化模型
        criterion = nn.CrossEntropyLoss()  # 定义损失函数
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # 定义优化器

        # 训练模型
        for epoch in range(10):  # 训练10轮
            model.train()  # 设置模型为训练模式
            for i, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()  # 清除梯度
                outputs = model(inputs)  # 前向传播计算输出
                loss = criterion(outputs, labels)  # 计算损失
                loss.backward()  # 反向传播计算梯度
                optimizer.step()  # 更新权重参数

            # 验证模型
            model.eval()  # 设置模型为评估模式
            all_labels = []
            all_preds = []
            correct = 0
            total = 0
            with torch.no_grad():  # 禁用梯度计算
                for inputs, labels in val_loader:
                    outputs = model(inputs)  # 前向传播计算输出
                    _, predicted = torch.max(outputs.data, 1)  # 获取预测值
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
    ' & {:.2f}'.format(average_accuracy * 100) + '$_{\pm' + '{:.2f}'.format(std_g_mean * 100)    + '}$'
)