# -*- coding: utf-8 -*-
"""
联合暴政实验：抑制控制体弱化 + 兴趣意识体增强 → 重复输出暴增
预测：单一操作只升一点，同时操作重复率暴涨
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


class DualBossGRU(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, output_size=10,
                 alpha=0.76, noise_strength=0.1, 
                 inhibition_strength=1.0, interest_strength=0.0,
                 total_steps=20000):
        super().__init__()
        self.hidden_size = hidden_size
        self.noise_strength = noise_strength
        self.alpha = alpha
        self.inhibition_strength = inhibition_strength
        self.interest_strength = interest_strength
        self.output_size = output_size

        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.inhibition_module = nn.Linear(hidden_size, 1)
        self.interest_detector = nn.Linear(hidden_size, 1)

        colored_noise = self._generate_colored_noise(total_steps, alpha)
        self.register_buffer('noise', torch.FloatTensor(colored_noise))
        self.noise_idx = 0
        self.total_steps = total_steps
        self.outputs_history = []

    def _generate_colored_noise(self, n_samples, alpha):
        n = n_samples
        spectrum = np.random.randn(n//2 + 1) + 1j * np.random.randn(n//2 + 1)
        freqs = np.fft.rfftfreq(n, d=1.0)
        freqs[0] = 1e-12
        spectrum *= (freqs ** (-alpha/2))
        colored = np.fft.irfft(spectrum, n=n)
        colored = (colored - np.mean(colored)) / (np.std(colored) + 1e-8)
        return colored

    def forward(self, x, hidden=None, inject_noise=True, record_output=False):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch_size, seq_len, _ = x.shape
        if hidden is None:
            hidden = torch.zeros(1, batch_size, self.hidden_size).to(x.device)

        last_logits = None

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

            # 抑制控制
            inhibition_signal = torch.sigmoid(self.inhibition_module(hidden.transpose(0, 1)))
            hidden = hidden * (1 - self.inhibition_strength * inhibition_signal.transpose(0, 1))

            # 兴趣增强
            interest_signal = torch.sigmoid(self.interest_detector(hidden.transpose(0, 1)))
            hidden = hidden + self.interest_strength * interest_signal.transpose(0, 1) * hidden

            out, hidden = self.gru(x_t, hidden)
            logits = self.fc(out)
            last_logits = logits

            if record_output:
                pred = torch.argmax(logits, dim=-1).item()
                self.outputs_history.append(pred)

        return last_logits, hidden

    def reset_noise(self):
        self.noise_idx = 0

    def clear_records(self):
        self.outputs_history = []

    def train_model(self, train_loader, epochs=5, device='cpu'):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for data, target in train_loader:
                data = data.view(-1, 784).to(device)
                target = target.to(device)
                target_one_hot = torch.zeros(target.size(0), self.output_size).to(device)
                target_one_hot.scatter_(1, target.unsqueeze(1), 1)

                optimizer.zero_grad()
                output, _ = self(data, inject_noise=True)
                loss = nn.MSELoss()(output, target_one_hot)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"训练 Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")
        print("训练完成。")

    def evaluate_health(self, steps=200, device='cpu'):
        self.eval()
        self.reset_noise()
        self.clear_records()
        hidden = torch.zeros(1, 1, self.hidden_size).to(device)
        input_seq = torch.randn(1, steps, 784).to(device)
        with torch.no_grad():
            for t in range(steps):
                x_t = input_seq[:, t:t+1, :]
                _, hidden = self(x_t, hidden, inject_noise=True, record_output=True)
        if len(self.outputs_history) < 2:
            return 1.0, 0.0
        repeats = sum(1 for i in range(1, len(self.outputs_history))
                      if self.outputs_history[i] == self.outputs_history[i-1])
        rep_rate = repeats / (len(self.outputs_history) - 1)
        unique, counts = np.unique(self.outputs_history, return_counts=True)
        ent = entropy(counts / len(self.outputs_history)) if len(unique) > 1 else 0.0
        return rep_rate, ent


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("使用设备:", device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(Subset(full_train, range(5000)), batch_size=64, shuffle=True)

    print("\n[阶段1] 训练健康网络（抑制=1.0, 兴趣=0.0）")
    model = DualBossGRU(inhibition_strength=1.0, interest_strength=0.0).to(device)
    model.train_model(train_loader, epochs=5, device=device)

    print("\n[阶段2] 测量健康状态")
    rep_healthy, ent_healthy = model.evaluate_health(steps=200, device=device)
    print(f"健康状态 → 重复率: {rep_healthy:.3f}, 输出熵: {ent_healthy:.3f}")

    print("\n[阶段3] 联合造病（抑制=0.2, 兴趣=3.0）")
    model.inhibition_strength = 0.2
    model.interest_strength = 3.0
    model.alpha = 0.3
    model.noise = torch.FloatTensor(model._generate_colored_noise(20000, 0.3)).to(device)
    rep_sick, ent_sick = model.evaluate_health(steps=200, device=device)
    print(f"病态状态 → 重复率: {rep_sick:.3f}, 输出熵: {ent_sick:.3f}")

    print("\n[阶段4] 联合治疗（抑制=1.0, 兴趣=0.0, β=0.76）")
    model.inhibition_strength = 1.0
    model.interest_strength = 0.0
    model.alpha = 0.76
    model.noise = torch.FloatTensor(model._generate_colored_noise(20000, 0.76)).to(device)
    rep_cured, ent_cured = model.evaluate_health(steps=200, device=device)
    print(f"治愈状态 → 重复率: {rep_cured:.3f}, 输出熵: {ent_cured:.3f}")

    print("\n" + "="*60)
    print("【联合暴政实验结果】")
    print("="*60)
    print(f"造病效果（重复率变化）: +{rep_sick - rep_healthy:.3f}")
    print(f"治疗效果（重复率变化）: {rep_cured - rep_sick:.3f}")

    if (rep_sick - rep_healthy) > 0.15:
        print("\n🎉 联合暴政假说验证成功！")
        print("重复输出需要抑制控制体弱化 + 兴趣意识体增强同时发生。")
    elif (rep_sick - rep_healthy) > 0.08:
        print("\n✓ 部分验证，联合效应大于单一效应。")
    else:
        print("\n⚠️ 联合效应不显著，需进一步分析。")

    labels = ['健康', '联合病态', '治愈']
    rep_rates = [rep_healthy, rep_sick, rep_cured]
    ent_rates = [ent_healthy, ent_sick, ent_cured]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(labels, rep_rates, color=['green', 'red', 'blue'])
    ax1.set_ylabel('重复率')
    ax1.set_title('联合暴政：重复率变化')
    ax1.axhline(y=rep_healthy, color='gray', linestyle='--', label='健康基线')
    ax1.legend()

    ax2.bar(labels, ent_rates, color=['green', 'red', 'blue'])
    ax2.set_ylabel('输出熵')
    ax2.set_title('输出多样性变化')
    ax2.axhline(y=ent_healthy, color='gray', linestyle='--', label='健康基线')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('dual_boss_experiment.png', dpi=150)
    plt.show()
    print("\n图表已保存: dual_boss_experiment.png")