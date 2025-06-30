import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class MambaBlock(nn.Module):
    """简化版Mamba块实现"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, use_conv=False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        self.use_conv = use_conv
        
        # 输入投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        
        # 可选的卷积层
        if self.use_conv:
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                kernel_size=d_conv,
                padding=d_conv - 1,
                groups=self.d_inner
            )
        
        # SSM参数
        self.x_proj = nn.Linear(self.d_inner, d_state * 2)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        # 激活函数
        self.activation = nn.SiLU()
        
        # 初始化SSM参数
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        # 输入投影
        xz = self.in_proj(x)  # (batch_size, seq_len, d_inner * 2)
        x_proj, z = xz.chunk(2, dim=-1)  # 每个 (batch_size, seq_len, d_inner)
        
        # 可选的卷积处理
        if self.use_conv:
            x_conv = self.conv1d(x_proj.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            x_conv = self.activation(x_conv)
        else:
            x_conv = self.activation(x_proj)
        
        # SSM
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # 计算 B, C, dt
        x_dbl = self.x_proj(x_conv)  # (batch_size, seq_len, d_state * 2)
        B, C = x_dbl.chunk(2, dim=-1)  # 每个 (batch_size, seq_len, d_state)
        
        dt = self.dt_proj(x_conv)  # (batch_size, seq_len, d_inner)
        dt = F.softplus(dt)
        
        # 简化的SSM计算
        y = self.selective_scan(x_conv, dt, A, B, C, self.D)
        
        # 门控
        y = y * self.activation(z)
        
        # 输出投影
        output = self.out_proj(y)
        
        return output
    
    def selective_scan(self, u, delta, A, B, C, D):
        # 简化实现：使用RNN风格的递归计算
        batch_size, seq_len, d_inner = u.shape
        d_state = A.shape[-1]
        
        # 初始化隐状态
        h = torch.zeros(batch_size, d_inner, d_state, device=u.device, dtype=u.dtype)
        
        outputs = []
        for i in range(seq_len):
            # 当前时间步的输入
            u_i = u[:, i, :]  # (batch_size, d_inner)
            delta_i = delta[:, i, :].unsqueeze(-1)  # (batch_size, d_inner, 1)
            B_i = B[:, i, :].unsqueeze(1)  # (batch_size, 1, d_state)
            C_i = C[:, i, :].unsqueeze(1)  # (batch_size, 1, d_state)
            
            # 状态更新
            dA = torch.exp(delta_i * A.unsqueeze(0))  # (batch_size, d_inner, d_state)
            dB = delta_i * B_i  # (batch_size, d_inner, d_state)
            
            h = h * dA + dB * u_i.unsqueeze(-1)  # (batch_size, d_inner, d_state)
            
            # 输出计算
            y_i = torch.sum(h * C_i, dim=-1) + D * u_i  # (batch_size, d_inner)
            outputs.append(y_i)
        
        return torch.stack(outputs, dim=1)  # (batch_size, seq_len, d_inner)

class MambaModel(nn.Module):
    """完整的Mamba模型"""
    def __init__(self, input_dim, d_model=256, n_layers=4, n_params=4, use_conv=False):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_params = n_params
        
        # 输入嵌入层
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Mamba层
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model, use_conv=use_conv) for _ in range(n_layers)
        ])
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_params),
            nn.Sigmoid()  # 确保输出在0-1之间
        )
        
    def forward(self, x, lengths=None):
        # x shape: (batch_size, max_seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # 输入嵌入
        x = self.input_embedding(x)  # (batch_size, seq_len, d_model)
        
        # 通过Mamba层
        for mamba_layer, layer_norm in zip(self.mamba_layers, self.layer_norms):
            residual = x
            x = mamba_layer(x)
            x = layer_norm(x + residual)
        
        # 全局平均池化（处理变长序列）
        if lengths is not None:
            # 创建掩码
            mask = torch.arange(seq_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
            
            # 应用掩码并计算平均
            x_masked = x * mask
            x_pooled = x_masked.sum(dim=1) / lengths.unsqueeze(1).float()
        else:
            x_pooled = x.mean(dim=1)  # (batch_size, d_model)
        
        # 输出预测
        params = self.output_layer(x_pooled)  # (batch_size, n_params)
        
        return params

class FinancialDataset(Dataset):
    """财务数据集"""
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        features = torch.FloatTensor(sample['features'])
        targets = torch.FloatTensor(sample['targets'])
        length = torch.LongTensor([len(sample['features'])])
        
        return features, targets, length

def collate_fn(batch):
    """处理变长序列的批处理函数"""
    features_list, targets_list, lengths_list = zip(*batch)
    
    # 获取最大长度
    max_length = max([f.shape[0] for f in features_list])
    feature_dim = features_list[0].shape[1]
    
    # 填充序列
    padded_features = torch.zeros(len(batch), max_length, feature_dim)
    targets = torch.stack(targets_list)
    lengths = torch.cat(lengths_list)
    
    for i, features in enumerate(features_list):
        seq_len = features.shape[0]
        padded_features[i, :seq_len, :] = features
    
    return padded_features, targets, lengths

def generate_synthetic_data(n_companies=100, min_quarters=8, max_quarters=40):
    """生成合成财务数据"""
    data = []
    
    for company_id in range(n_companies):
        # 随机选择季度数量
        n_quarters = random.randint(min_quarters, max_quarters)
        
        # 生成基础财务指标（随机游走）
        revenue = 1000 + np.cumsum(np.random.randn(n_quarters) * 50)
        profit = revenue * (0.1 + np.random.randn(n_quarters) * 0.02)
        assets = revenue * (2 + np.random.randn(n_quarters) * 0.1)
        equity = assets * (0.3 + np.random.randn(n_quarters) * 0.05)
        debt = assets - equity
        
        # 计算衍生指标
        profit_margin = profit / revenue
        roe = profit / equity
        debt_ratio = debt / assets
        asset_turnover = revenue / assets
        
        # 生成股价（与基本面相关但有噪声）
        stock_price = 50 + profit_margin * 100 + roe * 50 + np.random.randn(n_quarters) * 5
        stock_price = np.maximum(stock_price, 1)  # 确保股价为正
        
        # 组合特征
        features = np.column_stack([
            revenue / 1000,  # 标准化
            profit / 100,
            assets / 1000,
            equity / 1000,
            debt / 1000,
            profit_margin,
            roe,
            debt_ratio,
            asset_turnover,
            stock_price / 100
        ])
        
        # 生成目标参数（基于财务特征的复杂函数）
        # 这里模拟你的预测模型需要的4个参数
        param1 = 0.3 + 0.4 * np.tanh(np.mean(profit_margin))  # 盈利能力参数
        param2 = 0.2 + 0.6 * (1 / (1 + np.mean(debt_ratio)))  # 财务稳健性参数
        param3 = 0.1 + 0.8 * np.tanh(np.mean(roe))  # 增长潜力参数
        param4 = 0.4 + 0.3 * np.tanh(np.mean(asset_turnover) - 0.5)  # 运营效率参数
        
        targets = np.array([param1, param2, param3, param4])
        
        data.append({
            'company_id': company_id,
            'features': features,
            'targets': targets
        })
    
    return data

def train_model(use_conv=False):
    """训练模型"""
    # 生成数据
    print("生成合成数据...")
    data = generate_synthetic_data(n_companies=1000)
    
    # 数据标准化
    all_features = np.vstack([d['features'] for d in data])
    scaler = StandardScaler()
    scaler.fit(all_features)
    
    # 应用标准化
    for d in data:
        d['features'] = scaler.transform(d['features'])
    
    # 分割数据集
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    # 创建数据加载器
    train_dataset = FinancialDataset(train_data)
    val_dataset = FinancialDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # 创建模型
    input_dim = 10  # 财务特征维度
    model = MambaModel(input_dim=input_dim, d_model=128, n_layers=3, n_params=4, use_conv=use_conv)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # 训练循环
    print(f"开始训练 (使用卷积: {use_conv})...")
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(100):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_features, batch_targets, batch_lengths in train_loader:
            optimizer.zero_grad()
            
            outputs = model(batch_features, batch_lengths)
            loss = criterion(outputs, batch_targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_features, batch_targets, batch_lengths in val_loader:
                outputs = model(batch_features, batch_lengths)
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_mamba_model.pth')
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_mamba_model.pth'))
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss (Conv: {use_conv})')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 评估模型
    print("\n评估模型性能...")
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_features, batch_targets, batch_lengths in val_loader:
            outputs = model(batch_features, batch_lengths)
            all_predictions.append(outputs.numpy())
            all_targets.append(batch_targets.numpy())
    
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    
    # 计算每个参数的MAE和MAPE
    param_names = ['盈利能力', '财务稳健性', '增长潜力', '运营效率']
    
    print("\n各参数预测误差:")
    for i in range(4):
        mae = np.mean(np.abs(predictions[:, i] - targets[:, i]))
        mape = np.mean(np.abs((predictions[:, i] - targets[:, i]) / targets[:, i])) * 100
        print(f"{param_names[i]}: MAE={mae:.6f}, MAPE={mape:.2f}%")
    
    # 可视化预测结果
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i in range(4):
        axes[i].scatter(targets[:, i], predictions[:, i], alpha=0.6)
        axes[i].plot([0, 1], [0, 1], 'r--', lw=2)
        axes[i].set_xlabel(f'真实{param_names[i]}')
        axes[i].set_ylabel(f'预测{param_names[i]}')
        axes[i].set_title(f'{param_names[i]}预测效果')
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return model, scaler

def predict_new_company(model, scaler, quarters_data):
    """为新公司预测参数"""
    # 标准化输入数据
    normalized_data = scaler.transform(quarters_data)
    
    # 转换为张量
    features = torch.FloatTensor(normalized_data).unsqueeze(0)  # 添加batch维度
    length = torch.LongTensor([len(quarters_data)])
    
    # 预测
    model.eval()
    with torch.no_grad():
        params = model(features, length)
    
    return params.squeeze().numpy()

# 演示使用
if __name__ == "__main__":
    # 可以选择是否使用卷积
    USE_CONV = False  # 设置为False可以禁用卷积
    
    # 训练模型
    model, scaler = train_model(use_conv=USE_CONV)
    
    # 演示预测
    print("\n\n=== 预测演示 ===")
    
    # 创建一个示例公司的季度数据
    sample_quarters = np.array([
        # [收入, 利润, 资产, 股本, 债务, 利润率, ROE, 负债率, 资产周转率, 股价]
        [1200, 120, 2400, 720, 1680, 0.10, 0.167, 0.70, 0.50, 45],
        [1250, 135, 2500, 750, 1750, 0.108, 0.18, 0.70, 0.50, 48],
        [1300, 140, 2600, 780, 1820, 0.108, 0.179, 0.70, 0.50, 50],
        [1350, 150, 2700, 810, 1890, 0.111, 0.185, 0.70, 0.50, 52],
    ]) / np.array([1000, 100, 1000, 1000, 1000, 1, 1, 1, 1, 100])  # 标准化
    
    predicted_params = predict_new_company(model, scaler, sample_quarters)
    
    param_names = ['盈利能力', '财务稳健性', '增长潜力', '运营效率']
    print("预测的模型参数:")
    for name, param in zip(param_names, predicted_params):
        print(f"{name}: {param:.6f}")