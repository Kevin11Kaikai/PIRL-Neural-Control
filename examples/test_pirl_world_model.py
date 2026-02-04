"""
Test script for PIRL World Model

测试物理信息残差学习世界模型
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.world_model import test_world_model

if __name__ == "__main__":
    test_world_model()
