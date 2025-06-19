#!/usr/bin/env python3
"""
快速运行示例脚本
"""

import os
import sys
import subprocess
from pathlib import Path

def run_setup():
    """运行项目设置"""
    print("正在设置项目...")
    try:
        import setup_project
        setup_project.main()
        return True
    except Exception as e:
        print(f"设置失败: {e}")
        return False

def test_system():
    """测试系统"""
    print("\n正在测试系统...")
    try:
        result = subprocess.run([sys.executable, 'test_imports.py'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ 系统测试通过")
            return True
        else:
            print(f"✗ 系统测试失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"测试失败: {e}")
        return False

def run_quick_simulation():
    """运行快速模拟"""
    print("\n正在运行快速模拟...")
    
    # 创建简化的配置
    config = {
        "m_areas": 10,
        "L_energy_levels": 5,
        "T_periods": 24,
        "num_taxis": 100,
        "num_stations": 5,
        "simulation_duration": 120,
        "solver_method": "heuristic",
        "use_sample": True,
        "sample_size": 1000
    }
    
    # 保存临时配置
    import json
    config_file = 'temp_config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    try:
        # 运行模拟
        cmd = [sys.executable, 'main.py', '--config', config_file, '--output', 'quick_test']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # 清理临时文件
        if os.path.exists(config_file):
            os.remove(config_file)
        
        if result.returncode == 0:
            print("✓ 快速模拟完成")
            print("结果保存在 quick_test_* 目录中")
            return True
        else:
            print(f"✗ 模拟失败: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ 模拟超时")
        return False
    except Exception as e:
        print(f"✗ 模拟出错: {e}")
        return False

def main():
    """主函数"""
    print("电动出租车电池交换系统 - 快速示例")
    print("=" * 50)
    
    # 检查是否首次运行
    if not Path('dataprocess').exists():
        print("检测到首次运行，正在设置项目...")
        if not run_setup():
            print("项目设置失败！")
            return False
    
    # 测试系统
    if not test_system():
        print("系统测试失败！请检查环境和依赖。")
        return False
    
    # 询问用户是否运行快速模拟
    response = input("\n是否运行快速模拟示例？ (y/N): ").strip().lower()
    if response in ['y', 'yes', '是']:
        if run_quick_simulation():
            print("\n✓ 快速示例运行成功！")
            print("\n下一步:")
            print("1. 查看结果目录中的图表和报告")
            print("2. 修改配置文件运行完整模拟: python main.py --config configs/default_config.json")
            print("3. 使用真实NYC数据替换示例数据")
        else:
            print("\n✗ 快速示例运行失败")
    else:
        print("\n跳过快速模拟。")
        print("可以手动运行: python main.py --help")
    
    print("\n" + "=" * 50)
    print("设置完成！查看 QUICKSTART.md 了解详细使用方法。")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n用户中断。")
        sys.exit(1)
    except Exception as e:
        print(f"\n发生意外错误: {e}")
        sys.exit(1)