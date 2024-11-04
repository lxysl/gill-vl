import os
import pandas as pd
from pathlib import Path

# 读取原始tsv文件
df = pd.read_csv('datasets/cc3m_train.tsv', sep='\t')

# 获取特征文件目录下的所有文件
feature_dir = os.path.expanduser('~/Downloads/cc3m/training/hydit_clip_embs')
feature_files = os.listdir(feature_dir)

# 获取图片名列表（去掉.npy后缀）
valid_images = [f.replace('.npy', '') for f in feature_files if f.endswith('.npy')]

# 过滤数据框，只保留有特征的图片
filtered_df = df[df['image'].isin(valid_images)]

# 保存新的tsv文件
output_file = 'datasets/cc3m_train_sampled.tsv'
filtered_df.to_csv(output_file, sep='\t', index=False)

print(f'原始文件中的图片数量: {len(df)}')
print(f'过滤后的图片数量: {len(filtered_df)}')
print(f'新文件已保存至: {output_file}')