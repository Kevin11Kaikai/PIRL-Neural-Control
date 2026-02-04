"""
Test script for baseline controllers

测试传统控制器基线
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.agents.baselines import test_baseline_controllers

if __name__ == "__main__":
    test_baseline_controllers()
