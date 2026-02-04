"""
Test script for Neural CDE State Observer

测试基于 CDE 的状态观测器
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.models.neural_cde_observer import test_neural_cde_observer
    test_neural_cde_observer()
except ImportError as e:
    print(f"Error: {e}")
    print("\nPlease install torchcde:")
    print("  pip install torchcde")
