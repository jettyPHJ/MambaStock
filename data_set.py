import pandas as pd
import random
import numpy as np

file_path = '机器学习数据源.xlsx'
company_names = ['英伟达', '苹果']
feature_columns = ['营业额', '毛利率','净利率','经营现金流/营业额','平均股价','股价低点','股价高点','前期年营业额','本期年营业额','前期年毛利率',
                  '本期年毛利率','前期年净利润','本期年净利润']

def get_index(feature_name):
    return feature_columns.index(feature_name)

# 读取excel数据
def load_excel(): 
    data_map = {}
    for company_name in company_names:
        df = pd.read_excel(file_path, sheet_name=company_name)
        
        # 提取存在的列
        data_map[company_name] = {}
        for col in feature_columns:
            if col in df.columns:
                data_map[company_name][col] = df[col].to_numpy()
    return data_map

def generate_synthetic_data(source:dict):
    """生成训练的财务数据"""
    data = []
    
    for _, company_data in source.items():
        row_count = len(next(iter(company_data.values())))
        
        # 至少要六行数据才能构造训练样本（否则无法避免包含最后一行）
        if row_count < 6:
            continue
        
        # 每个公司生成的样本数量为 row_count // 2
        sample_size = row_count // 2
        
        feature_names = list(company_data.keys())

        for _ in range(sample_size):
            # 随机决定子序列长度，范围为 [5, row_count - 1]
            seq_len = random.randint(5, row_count-1)
            
            # 随机选择起点，使得子序列不包含最后一行
            max_start = row_count - seq_len - 1
            start_idx = random.randint(0, max_start)

            # 提取每一列在指定范围的数据
            raw_data = []
            sample = []
            target = 0.0
            for col_name in feature_names:               
                # 提取子序列
                col_data = company_data[col_name][start_idx : start_idx + seq_len]
                col_arr = np.array(col_data, dtype=float) 
                # 如果是目标列 '平均股价'，在同样的 min-max 体系下计算 target
                if col_name == '平均股价':
                    next_price = company_data[col_name][start_idx + seq_len]
                    target = next_price  / col_data[-1]
                # 对原始数据使用 Sigmoid 归一化
                raw_data.append(col_data)
                col_norm = sigmoid_normalize(col_arr) 
                sample.append(col_norm)
             # 组装 [seq_len, feature_dim]
            orin_data = np.stack(raw_data, axis=1)
            features = np.stack(sample, axis=1)
            data.append({
                'origin': orin_data,
                'features': features,
                'target': target
            })
            
    return data

def sigmoid_normalize(arr, scale=8.0):
    arr = np.array(arr, dtype=float)
    min_val, max_val = np.min(arr), np.max(arr)
    scaled = (arr - min_val) / (max_val - min_val + 1e-8)
    centered = (scaled - 0.5) * scale
    return 1 / (1 + np.exp(-centered))

raw_data = load_excel()
data_source = generate_synthetic_data(raw_data)