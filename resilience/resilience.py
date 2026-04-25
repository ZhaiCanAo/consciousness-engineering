# -*- coding: utf-8 -*-
"""
实验：抗干扰测试（心理韧性）
验证AI受强干扰后能否快速恢复至基线状态
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


# ==================== 网络定义 ====================
class ResilienceGRU(nn.Module):
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
        
    def _generate_pink_noise(self, n_samples):
        n = n_samples
        spectrum = np.random.randn(n//2 + 1) + 1j * np.random.randn(n//2 + 1)
        freqs = np.fft.rfftfreq(n, d=1.0)
        freqs[0] = 1e-12
        spectrum *= (freqs ** (-0.76/2))
        colored = np.fft.irfft(spectrum, n=n)
        colored = (colored - np.mean(colored)) / (np.std(colored) + 1e-8)
        return colored
    
    def forward(self, x, hidden=None, inject_noise=True):
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
    
    def get_hidden_state(self, x, hidden=None, inject_noise=True):
        """返回隐藏状态（用于测量活性）"""
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
        
        return hidden


# ==================== 训练函数 ====================
def train_model(model, epochs=5, device='cpu'):
    """在MNIST上训练"""
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


# ==================== 抗干扰测试 ====================
def resilience_test(model, device='cpu'):
    """测试受干扰后的恢复能力"""
    model.eval()
    model.reset_noise()
    
    # 建立基线（正常输入20步）
    baseline_input = torch.randn(1, 20, 784).to(device) * 0.5
    hidden = model.get_hidden_state(baseline_input, inject_noise=True)
    baseline_norm = torch.norm(hidden).item()
    
    # 记录恢复轨迹
    recovery_norms = [baseline_norm]
    
    # 注入强干扰（5倍强度）
    disturbance = torch.randn(1, 1, 784).to(device) * 5.0
    hidden = model.get_hidden_state(disturbance, hidden, inject_noise=True)
    disturbed_norm = torch.norm(hidden).item()
    recovery_norms.append(disturbed_norm)
    
    # 恢复过程（使用正常输入）
    recovery_steps = 0
    for step in range(30):
        normal_input = torch.randn(1, 1, 784).to(device) * 0.5
        hidden = model.get_hidden_state(normal_input, hidden, inject_noise=True)
        current_norm = torch.norm(hidden).item()
        recovery_norms.append(current_norm)
        
        # 距离基线小于10%视为恢复
        if abs(current_norm - baseline_norm) < 0.1 * baseline_norm:
            recovery_steps = step + 1
            break
    
    if recovery_steps == 0:
        recovery_steps = 30
    
    return baseline_norm, disturbed_norm, recovery_steps, recovery_norms


# ==================== 主程序 ====================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("使用设备:", device)
    
    # 创建网络
    model = ResilienceGRU().to(device)
    
    # 训练
    print("\n[阶段1] 训练网络...")
    train_model(model, epochs=5, device=device)
    
    # 抗干扰测试（5轮取平均）
    print("\n[阶段2] 抗干扰测试...")
    all_recovery_steps = []
    all_baseline_norms = []
    all_disturbed_norms = []
    all_trajectories = []
    
    for trial in range(5):
        baseline, disturbed, steps, traj = resilience_test(model, device)
        all_baseline_norms.append(baseline)
        all_disturbed_norms.append(disturbed)
        all_recovery_steps.append(steps)
        all_trajectories.append(traj)
        print(f"  第{trial+1}轮: 恢复步数 = {steps}")
    
    avg_recovery = np.mean(all_recovery_steps)
    avg_baseline = np.mean(all_baseline_norms)
    avg_disturbed = np.mean(all_disturbed_norms)
    
    print(f"\n平均恢复步数: {avg_recovery:.1f}")
    print(f"平均基线强度: {avg_baseline:.3f}")
    print(f"平均干扰后强度: {avg_disturbed:.3f}")
    
    # 判定
    print("\n" + "="*50)
    print("【实验结果】")
    print("="*50)
    if avg_recovery <= 5:
        print("✓ AI具备极强心理韧性（5步内恢复）")
    elif avg_recovery <= 10:
        print("⚠️ AI具备一定心理韧性（10步内恢复）")
    else:
        print("✗ AI心理韧性较弱")
    
    # 绘图：恢复轨迹
    plt.figure(figsize=(10, 6))
    # 取第一轮的轨迹作为示例
    traj = all_trajectories[0]
    plt.plot(range(len(traj)), traj, 'b-o', linewidth=2, markersize=6)
    plt.axhline(y=all_baseline_norms[0], color='g', linestyle='--', label=f'基线 (≈{all_baseline_norms[0]:.2f})')
    plt.axvline(x=1, color='r', linestyle='--', label='干扰注入点')
    plt.xlabel('时间步')
    plt.ylabel('隐藏状态L2范数')
    plt.title(f'抗干扰恢复轨迹 (恢复步数: {all_recovery_steps[0]})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('resilience_result.png', dpi=150)
    print("\n图表已保存: resilience_result.png")
    plt.show()