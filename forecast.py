import torch
import pandas as pd
import numpy as np
from MambaStock import MambaModel
import data_set

file_path = '测试数据.xlsx'
model_path = 'best_mamba_model.pth'

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取特征维度
company_names, feature_columns = data_set.get_excel_meta(file_path,exclude_columns = data_set.exclude_columns)

# 验证测试数据和训练数据的feature_columns是否相同
if feature_columns != data_set.feature_columns:
    raise ValueError("测试数据和训练数据的特征列不匹配。")

input_dim = len(feature_columns)
# 创建并加载模型
model = MambaModel()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 读取数据
data_map = data_set.load_excel(file_path, company_names, feature_columns)

# 输出结果字典
df_results = {}
min_size = data_set.min_sample_size - 1

# 遍历每个公司进行预测
for company in company_names:
    company_data = data_map[company]
    num_rows = len(next(iter(company_data.values())))
    if num_rows < min_size:
        print(company, ":样本不足, 无法预测")
        continue

    preds = [None] * min_size
    for i in range(min_size, num_rows):
        sub_data = {key: val[:i+1] for key, val in company_data.items()}

        # 构造输入序列
        raw_seq = []
        norm_seq = []
        for col in feature_columns:
            arr = np.array(sub_data[col], dtype=float)
            raw_seq.append(arr)
            norm_arr = data_set.sigmoid_normalize(arr)
            norm_seq.append(norm_arr)

        origin_tensor = torch.tensor(np.stack(raw_seq, axis=1), dtype=torch.float32).to(device)
        feature_tensor = torch.tensor(np.stack(norm_seq, axis=1), dtype=torch.float32).to(device)

        origin_tensor = origin_tensor.unsqueeze(0)  # (1, seq_len, dim)
        feature_tensor = feature_tensor.unsqueeze(0)
        lengths = torch.tensor([feature_tensor.shape[1]], dtype=torch.long).to(device)

        with torch.no_grad():
            pred = model([origin_tensor[0]], feature_tensor, lengths)
            preds.append(pred.item())

    # 保存预测结果
    df_company = pd.DataFrame(company_data)
    df_company['模型预测结果'] = preds
    df_results[company] = df_company

# 直接写回原始文件
with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    for company, df in df_results.items():
        df.to_excel(writer, sheet_name=company, index=False)

print("所有公司预测完成，结果已写入测试数据.xlsx")