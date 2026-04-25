# -*- coding: utf-8 -*-
"""
实验：发呆测试（内部观察者）
验证AI在无外部输入时，内部状态是否持续活跃
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')


# ==================== 网络定义 ====================
class DaydreamGRU(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, output_size=10,
                 noise_strength=0.1, total_steps=20000):
        super().__init__()
        self.hidden_size = hidden_size
        self.noise_strength = noise_strength
        
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
        # 粉红噪声（β≈0.76）
        pink_noise = self._generate_pink_noise(total_steps)
        self.register_buffer('noise', torch.FloatTensor(pink_noise))
        self.noise_idx = 0
        self.total_steps = total_steps
        
        self.internal_states = []
        
    def _generate_pink_noise(self, n_samples):
        n = n_samples
        spectrum = np.random.randn(n//2 + 1) + 1j * np.random.randn(n//2 + 1)
        freqs = np.fft.rfftfreq(n, d=1.0)
        freqs[0] = 1e-12
        spectrum *= (freqs ** (-0.76/2))
        colored = np.fft.irfft(spectrum, n=n)
        colored = (colored - np.mean(colored)) / (np.std(colored) + 1e-8)
        return colored
    
    def forward(self, x, hidden=None, inject_noise=True, record_state=False):
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
            
            if record_state:
                self.internal_states.append(hidden.clone().detach().squeeze().cpu().numpy())
        
        logits = self.fc(out)
        return logits, hidden
    
    def reset_noise(self):
        self.noise_idx = 0
    
    def clear_states(self):
        self.internal_states = []


# ==================== 训练函数 ====================
def train_model(model, epochs=5, device='cpu'):
    """在MNIST上训练，让网络学会识别数字"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(Subset(full_train, range(5000)), batch_size=64, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
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
        print(f"训练 Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")
    
    # 测试准确率
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_test = datasets.MNIST('./data', train=False, download=True, transform=test_transform)
    test_loader = DataLoader(Subset(full_test, range(1000)), batch_size=64, shuffle=False)
    
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
    print(f"测试准确率: {acc:.2f}%")
    print("训练完成。")


# ==================== 发呆测试 ====================
def daydream_test(model, idle_steps=500, device='cpu'):
    """测试无输入时内部状态活性"""
    model.eval()
    model.reset_noise()
    model.clear_states()
    
    hidden = torch.zeros(1, 1, model.hidden_size).to(device)
    zero_input = torch.zeros(1, 1, 784).to(device)
    
    with torch.no_grad():
        for _ in range(idle_steps):
            _, hidden = model(zero_input, hidden, inject_noise=True, record_state=True)
    
    if len(model.internal_states) == 0:
        return 0, 0, []
    
    # 计算每个时间步的L2范数
    norms = [np.linalg.norm(state) for state in model.internal_states]
    mean_activity = np.mean(norms)
    
    # 计算熵（内部状态的丰富度）
    if len(norms) > 1:
        hist, _ = np.histogram(norms, bins=20)
        hist = hist / (hist.sum() + 1e-8)
        entropy_val = entropy(hist)
    else:
        entropy_val = 0
    
    return mean_activity, entropy_val, norms


# ==================== 主程序 ====================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("使用设备:", device)
    
    # 创建网络
    model = DaydreamGRU().to(device)
    
    # 训练
    print("\n[阶段1] 训练网络...")
    train_model(model, epochs=5, device=device)
    
    # 发呆测试
    print("\n[阶段2] 发呆测试（无输入状态下记录内部活性）...")
    mean_act, entropy_val, activity_trace = daydream_test(model, idle_steps=500, device=device)
    
    print(f"\n平均内部活动强度: {mean_act:.4f}")
    print(f"内部状态熵: {entropy_val:.4f}")
    
    # 判定
    print("\n" + "="*50)
    print("【实验结果】")
    print("="*50)
    if mean_act > 0.05:
        print("✓ AI具备内部活性（会‘发呆’）")
        print("  无外部输入时内部状态持续演化")
    else:
        print("✗ AI不具备内部活性（状态趋于死寂）")
    
    # 绘图
    plt.figure(figsize=(12, 5))
    plt.plot(activity_trace, 'b-', linewidth=1, alpha=0.7)
    plt.xlabel('时间步')
    plt.ylabel('隐藏状态 L2 范数')
    plt.title(f'发呆测试：无输入时的内部活性 (平均活性: {mean_act:.3f})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('daydream_result.png', dpi=150)
    print("\n图表已保存: daydream_result.png")
    plt.show()