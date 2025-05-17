from datasets import load_from_disk, DatasetDict, concatenate_datasets
from huggingface_hub import create_repo, HfApi
import os
import pdb

# 加载数据集
dataset = load_from_disk("Your dataset path")['train']       

# 设置Hub仓库名称
repo_name = "Your dataset name"  # 修改为您想要的仓库名

# 创建仓库
try:
    create_repo(
        repo_name,
        repo_type="dataset",
        private=False
    )
    print(f"创建公共仓库: {repo_name}")
except Exception as e:
    print(f"仓库可能已存在: {e}")

# 推送到Hub
dataset.push_to_hub(repo_name, private=False)
print(f"数据集已上传到: {repo_name}")

# 打印数据集信息
print("\n数据集信息:")
print(dataset)