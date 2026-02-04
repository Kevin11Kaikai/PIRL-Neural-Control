"""
Test script for PhIHP Agent

测试物理信息分层规划强化学习代理
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.agents.phihp_agent import test_phihp_agent

if __name__ == "__main__":
    test_phihp_agent()
