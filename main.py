import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import random
import data_set
import os

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
        _, seq_len, _ = x.shape
        
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
    def __init__(self, input_dim, d_model=128, n_layers=4, n_params=3, use_conv=False):
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
        
    def forward(self, origins, x, lengths=None):

        _, seq_len, _ = x.shape

        # 输入嵌入
        x = self.input_embedding(x)  # (batch_size, seq_len, d_model)

        # Mamba 层 + 残差连接
        for mamba_layer, layer_norm in zip(self.mamba_layers, self.layer_norms):
            residual = x
            x = mamba_layer(x)
            x = layer_norm(x + residual)

        # 池化：考虑 mask
        if lengths is not None:
            mask = torch.arange(seq_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)

            x_masked = x * mask
            valid_counts = mask.sum(dim=1) + 1e-8
            x_pooled = x_masked.sum(dim=1) / valid_counts
        else:
            x_pooled = x.mean(dim=1)

        # 输出参数预测
        params = self.output_layer(x_pooled)  # (batch_size, n_params)
        # 确保参数在合理范围内
        params = params * 2           # output ∈ (0, 3)
        # 自定义公式输出
        output = self.manual_financial_model(origins, params)
        return output
    
    def manual_financial_model(self, oringins, params):
        """
        使用自定义的金融公式计算每个序列样本的预测股价。
        - oringins: shape (batch_size, seq_len, input_dim)
        - params: shape (batch_size, n_params)
        返回: shape (batch_size,)
        """
        # 拆解参数
        a, b, c = params[:, 0], params[:, 1], params[:, 2]  # shape: (batch_size,)
        
        # 取每个序列的最后两个时间步
        last_1 = torch.stack([x[-1] for x in oringins], dim=0)  # shape: (batch_size, input_dim)
        last_2 = torch.stack([x[-2] for x in oringins], dim=0)  # shape: (batch_size, input_dim)
        ry_now , ry_before = last_1[:, data_set.get_index('TTM营业额')], last_2[:, data_set.get_index('TTM营业额')]
        gy_now , gy_before = last_1[:, data_set.get_index('TTM毛利率')], last_2[:, data_set.get_index('TTM毛利率')]
        ny_now , ny_before = last_1[:, data_set.get_index('TTM净利率')], last_2[:, data_set.get_index('TTM净利率')]
        
        # 构造手工股价预测公式
        output = a * ry_now / ry_before * 2 * (b * torch.sigmoid(gy_now - gy_before) + c * torch.sigmoid(ny_now - ny_before))      
                   
        return output  # (batch_size,)

class FinancialDataset(Dataset):
    """财务数据集"""
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        origins = torch.tensor(sample['origin'], dtype=torch.float32) # shape: (seq_len, input_dim)
        features = torch.tensor(sample['features'], dtype=torch.float32)  # shape: (seq_len, input_dim)
        target = torch.tensor([sample['target']], dtype=torch.float32)  # shape: (1,)
        return origins, features, target
def collate_fn(batch):
    """
    batch: List of tuples from Dataset.__getitem__:
        Each tuple: (origin: Tensor(seq_len, input_dim),
                     features: Tensor(seq_len, input_dim),
                     target: Tensor(()))
    """
    max_len = max(f.shape[0] for _, f, _ in batch)
    input_dim = batch[0][1].shape[1]  # features.shape[1]

    batch_origins = []
    batch_features = []
    batch_targets = []
    lengths = []

    for origins, features, targets in batch:
        lengths.append(features.shape[0])  # 保存真实长度
        pad_len = max_len - features.shape[0]
        if pad_len > 0:
            pad = torch.zeros(pad_len, input_dim)
            features = torch.cat([features, pad], dim=0)

        batch_origins.append(origins)
        batch_features.append(features)
        batch_targets.append(targets)
             
    batch_features = torch.stack(batch_features)            # (batch_size, max_seq_len, input_dim)
    batch_targets = torch.stack(batch_targets).squeeze()    # (batch_size,)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return batch_origins, batch_features, batch_targets, lengths



def train_model(model : MambaModel):
    """训练模型"""
    data = data_set.data_source    
    
    # 分割数据集
    random.shuffle(data)
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    # 创建数据加载器
    train_dataset = FinancialDataset(train_data)
    val_dataset = FinancialDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # 打印模型参数信息
    print(f"可训练参数总量: {count_parameters(model):,}")
    # print_model_parameters(model)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # 训练循环
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(100):
        # 训练阶段
        model.train()
        train_loss = 0
        for origins, batch_features, batch_targets, lengths in train_loader:

            origins = [o.to(device) for o in origins]  # origin 是 list[Tensor]，需要单独处理
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            
            outputs = model(origins, batch_features, lengths) # shape: (batch_size,)
            loss = criterion(outputs, batch_targets) # batch_targets: (batch_size,)
            
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
            for origins, batch_features, batch_targets, lengths in val_loader:

                origins = [o.to(device) for o in origins]
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                lengths = lengths.to(device)

                outputs = model(origins, batch_features, lengths)
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
        
        print(f'Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        plot_train_val_loss(train_losses, val_losses, save_path='logs/loss.png')
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break     
    # 加载训练中验证集表现最好的模型参数
    model.load_state_dict(torch.load('best_mamba_model.pth'))
    return model

#显示训练参数
def print_model_parameters(model):
    print("模型结构及参数：")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:50s} shape: {str(list(param.shape)):>20}  参数量: {param.numel():,}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 绘制训练曲线
def plot_train_val_loss(train_losses, val_losses, save_path='loss.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # 确保目录存在
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    # 保存并关闭
    plt.savefig(save_path)
    plt.close()
    return

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

    print(f"CUDA版本: {torch.version.cuda}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 可以选择是否使用卷积
    USE_CONV = False  # 设置为False可以禁用卷积
    print(f"开始训练 (使用卷积: {USE_CONV})...")

    # 创建模型
    feature_dim = len(data_set.feature_columns)  # 财务特征维度
    model = MambaModel(input_dim = feature_dim, d_model = 128, n_layers = 5, use_conv = USE_CONV)
    model = model.to(device)

    print(f"Using device: {device}")

    # 训练模型
    model = train_model(model)