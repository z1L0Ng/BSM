#!/usr/bin/env python3
"""
测试导入和基本功能的脚本
"""

import sys
import os
import traceback
from pathlib import Path

def test_imports():
    """测试所有模块的导入"""
    print("测试模块导入...")
    
    # 测试基本导入
    try:
        import numpy as np
        import pandas as pd
        print("✓ NumPy和Pandas导入成功")
    except ImportError as e:
        print(f"✗ NumPy/Pandas导入失败: {e}")
        return False
    
    # 测试数据处理模块
    try:
        from dataprocess.loaddata import load_trip_data, clean_trip_data
        from dataprocess.distance import create_all_matrices
        print("✓ 数据处理模块导入成功")
    except ImportError as e:
        print(f"✗ 数据处理模块导入失败: {e}")
        print("请确保dataprocess目录存在并包含__init__.py文件")
        return False
    
    # 测试模型模块
    try:
        from models.state_transition import ETaxiStateModel, BSSStateModel
        from models.optimization_model import JointOptimizationModel
        print("✓ 模型模块导入成功")
    except ImportError as e:
        print(f"✗ 模型模块导入失败: {e}")
        print("请确保models目录存在并包含__init__.py文件")
        return False
    
    # 测试优化模块
    try:
        from optimization.joint_optimizer import JointOptimizer, OptimizationConfig
        from optimization.charge_scheduler import ChargeTaskGenerator
        from optimization.interface import optimize_bss_layout
        print("✓ 优化模块导入成功")
    except ImportError as e:
        print(f"✗ 优化模块导入失败: {e}")
        print("请确保optimization目录存在并包含__init__.py文件")
        return False
    
    # 测试工具模块
    try:
        from utils.visualization import plot_station_metrics, plot_taxi_metrics
        from utils.analysis import analyze_performance
        print("✓ 工具模块导入成功")
    except ImportError as e:
        print(f"✗ 工具模块导入失败: {e}")
        print("请确保utils目录存在并包含__init__.py文件")
        return False
    
    return True

def create_missing_init_files():
    """创建缺失的__init__.py文件"""
    print("创建必要的__init__.py文件...")
    
    directories = [
        'dataprocess',
        'models', 
        'optimization',
        'utils',
        'agent',
        'bss',
        'environment',
        'scheduler',
        'simulation'
    ]
    
    for dir_name in directories:
        init_file = Path(dir_name) / '__init__.py'
        if not init_file.exists():
            init_file.parent.mkdir(exist_ok=True)
            init_file.touch()
            print(f"创建: {init_file}")
    
    print("__init__.py文件创建完成")

def test_basic_functionality():
    """测试基本功能"""
    print("\n测试基本功能...")
    
    try:
        # 测试状态转移模型
        from models.state_transition import ETaxiStateModel
        model = ETaxiStateModel(m_areas=5, L_energy_levels=10, T_periods=24)
        print("✓ 状态转移模型创建成功")
        
        # 测试优化配置
        from optimization.joint_optimizer import OptimizationConfig
        config = OptimizationConfig(
            m_areas=5,
            L_energy_levels=10,
            T_periods=24
        )
        print("✓ 优化配置创建成功")
        
        # 测试数据生成
        import numpy as np
        np.random.seed(42)
        
        # 创建测试区块位置
        block_positions = {i: (i*10, i*10) for i in range(5)}
        print("✓ 测试数据生成成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 基本功能测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("电动出租车电池交换系统 - 导入测试")
    print("=" * 50)
    
    # 添加当前目录到Python路径
    current_dir = Path.cwd()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    # 创建必要的__init__.py文件
    create_missing_init_files()
    
    # 测试导入
    if not test_imports():
        print("\n导入测试失败！请检查文件结构和依赖。")
        return False
    
    # 测试基本功能
    if not test_basic_functionality():
        print("\n基本功能测试失败！")
        return False
    
    print("\n" + "=" * 50)
    print("所有测试通过！系统准备就绪。")
    print("现在可以运行: python main.py --help")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)