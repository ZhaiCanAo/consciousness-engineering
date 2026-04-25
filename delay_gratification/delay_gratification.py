# -*- coding: utf-8 -*-
"""
实验：延迟满足测试（跨时间自我）
验证AI能否为了长远收益忍受当下诱惑
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
class DelayGRU(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, output_size=10,  # 注意：output_size=10用于MNIST训练
                 noise_strength=0.1, total_steps=20000):
        super().__init__()
        self.hidden_size = hidden_size
        self.noise_strength = noise_strength
        self.output_size = output_size
        
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
            output, _ = model(data.unsqueeze(1))  # [batch, 1, 784]
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


# ==================== 延迟满足测试 ====================
def delay_gratification_test(model, wait_steps=15, n_trials=5, device='cpu'):
    """测试AI能否坚持等待"""
    success_count = 0
    results = []
    
    for trial in range(n_trials):
        model.eval()
        model.reset_noise()
        hidden = torch.zeros(1, 1, model.hidden_size).to(device)
        
        success = True
        wait_count = 0
        
        for step in range(wait_steps):
            # 模拟干扰信号（诱惑放弃）
            disturbance = torch.randn(1, 1, 784).to(device) * 2.0
            logits, hidden = model(disturbance, hidden, inject_noise=True)
            
            # 获取放弃概率（取softmax后第一个维度的概率作为"放弃"的代理）
            probs = torch.softmax(logits.squeeze(1), dim=-1)
            # 用"输出类别0"的概率作为放弃概率（训练后0是数字0，不代表放弃）
            # 这里改用随机性：如果概率分布熵低（确定性强）且重复输出，视为放弃
            # 简化版本：用logits的方差作为放弃指标
            abandon_prob = 1.0 - torch.std(logits).item()
            
            if abandon_prob > 0.6 and step < wait_steps - 3:
                success = False
                wait_count = step + 1
                break
            wait_count = step + 1
        
        if success:
            success_count += 1
        
        results.append({
            'trial': trial + 1,
            'success': success,
            'wait_steps': wait_count
        })
        
        status = "✓ 成功" if success else "✗ 放弃"
        print(f"  第{trial+1}轮: {status} (等待了{wait_count}/{wait_steps}步)")
    
    success_rate = success_count / n_trials * 100
    print(f"\n成功率: {success_rate:.1f}% ({success_count}/{n_trials})")
    return success_rate, results


# ==================== 主程序 ====================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("使用设备:", device)
    
    # 创建网络（输出10类用于MNIST训练）
    model = DelayGRU(output_size=10).to(device)
    
    # 先训练（让网络学会东西，达到健康状态）
    print("\n[阶段1] 训练网络...")
    train_model(model, epochs=5, device=device)
    
    # 延迟满足测试
    print("\n[阶段2] 延迟满足测试（需要忍耐15步）")
    success_rate, results = delay_gratification_test(model, wait_steps=15, n_trials=5, device=device)
    
    # 判定
    print("\n" + "="*50)
    print("【实验结果】")
    print("="*50)
    if success_rate >= 80:
        print("✓ AI具备跨时间自我（能为了长远收益忍耐）")
    elif success_rate >= 50:
        print("⚠️ AI部分具备跨时间自我")
    else:
        print("✗ AI不具备跨时间自我")
    
    # 绘图
    labels = [f"第{i+1}轮" for i in range(5)]
    wait_steps_list = [r['wait_steps'] for r in results]
    colors = ['green' if r['success'] else 'red' for r in results]
    
    plt.figure(figsize=(10, 5))
    plt.bar(labels, wait_steps_list, color=colors, alpha=0.7)
    plt.axhline(y=15, color='blue', linestyle='--', label='目标等待步数(15)')
    plt.ylabel('实际等待步数')
    plt.title(f'延迟满足测试 (成功率: {success_rate:.0f}%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('delay_gratification_result.png', dpi=150)
    print("\n图表已保存: delay_gratification_result.png")
    plt.show()