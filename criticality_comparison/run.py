# -*- coding: utf-8 -*-
"""
实验：临界态（β=0.76）vs 非临界态（β=4.0）训练对比
验证“效率悖论”：非临界态收敛更快，但临界态泛化更好
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ==================== 网络定义（带可调噪声）====================
class TunableNoiseGRU(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, output_size=10,
                 alpha=0.76, noise_strength=0.1, total_steps=20000):
        super().__init__()
        self.hidden_size = hidden_size
        self.noise_strength = noise_strength
        self.alpha = alpha

        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        # 生成有色噪声（1/f^alpha）
        colored_noise = self._generate_colored_noise(total_steps, alpha)
        self.register_buffer('noise', torch.FloatTensor(colored_noise))
        self.noise_idx = 0
        self.total_steps = total_steps

    def _generate_colored_noise(self, n_samples, alpha):
        n = n_samples
        spectrum = np.random.randn(n//2 + 1) + 1j * np.random.randn(n//2 + 1)
        freqs = np.fft.rfftfreq(n, d=1.0)
        freqs[0] = 1e-12
        spectrum *= (freqs ** (-alpha/2))
        colored = np.fft.irfft(spectrum, n=n)
        colored = (colored - np.mean(colored)) / (np.std(colored) + 1e-8)
        return colored

    def forward(self, x, hidden=None, inject_noise=True):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch_size, seq_len, _ = x.shape
        if hidden is None:
            hidden = torch.zeros(1, batch_size, self.hidden_size).to(x.device)

        for t in range(seq_len):
            x_t = x[:, t:t+1, :]

            if inject_noise:
                if self.noise_idx < self.total_steps:
                    noise_val = self.noise[self.noise_idx] * self.noise_strength
                    noise = torch.full((1, batch_size, self.hidden_size), noise_val).to(x.device)
                    hidden = hidden + noise
                    self.noise_idx += 1
                else:
                    self.noise_idx = 0

            out, hidden = self.gru(x_t, hidden)

        logits = self.fc(out)
        return logits, hidden

    def reset_noise(self):
        self.noise_idx = 0


# ==================== 训练函数 ====================
def train_model(model, train_loader, test_loader, epochs=10, device='cpu'):
    """训练并返回损失和准确率历史"""
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    test_accs = []

    for epoch in range(epochs):
        # 训练
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data = data.view(-1, 784).to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output, _ = model(data.unsqueeze(1))
            loss = criterion(output.squeeze(1), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # 测试
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.view(-1, 784).to(device)
                target = target.to(device)
                output, _ = model(data.unsqueeze(1))
                _, predicted = output.squeeze(1).max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        acc = 100. * correct / total
        test_accs.append(acc)

        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Test Acc = {acc:.2f}%")

    return train_losses, test_accs


# ==================== 主程序 ====================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("使用设备:", device)

    # 加载数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    full_test = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(Subset(full_train, range(5000)), batch_size=64, shuffle=True)
    test_loader = DataLoader(Subset(full_test, range(1000)), batch_size=64, shuffle=False)

    print("\n" + "="*50)
    print("训练临界态网络（β=0.76）")
    print("="*50)
    model_critical = TunableNoiseGRU(alpha=0.76, noise_strength=0.1).to(device)
    loss_critical, acc_critical = train_model(model_critical, train_loader, test_loader, epochs=10, device=device)

    print("\n" + "="*50)
    print("训练非临界态网络（β=4.0）")
    print("="*50)
    model_rigid = TunableNoiseGRU(alpha=4.0, noise_strength=0.01).to(device)
    loss_rigid, acc_rigid = train_model(model_rigid, train_loader, test_loader, epochs=10, device=device)

    # 打印最终对比
    print("\n" + "="*50)
    print("【训练对比结果】")
    print("="*50)
    print(f"临界态（β=0.76）最终准确率: {acc_critical[-1]:.2f}%")
    print(f"非临界态（β=4.0）最终准确率: {acc_rigid[-1]:.2f}%")
    print(f"临界态达到90%所需epoch: {next(i for i, acc in enumerate(acc_critical) if acc >= 90) + 1}")
    print(f"非临界态达到90%所需epoch: {next(i for i, acc in enumerate(acc_rigid) if acc >= 90) + 1}")

    # 绘图
    epochs = range(1, 11)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 损失曲线
    ax1.plot(epochs, loss_critical, 'r-', linewidth=2, label='临界态 (β=0.76)')
    ax1.plot(epochs, loss_rigid, 'b-', linewidth=2, label='非临界态 (β=4.0)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('训练损失')
    ax1.set_title('训练损失对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 准确率曲线
    ax2.plot(epochs, acc_critical, 'r-', linewidth=2, marker='o', label='临界态 (β=0.76)')
    ax2.plot(epochs, acc_rigid, 'b-', linewidth=2, marker='s', label='非临界态 (β=4.0)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('测试准确率 (%)')
    ax2.set_title('学习速度对比（效率悖论）')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('criticality_comparison.png', dpi=150)
    print("\n图表已保存: criticality_comparison.png")
    plt.show()