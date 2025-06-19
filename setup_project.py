#!/usr/bin/env python3
"""
项目设置脚本 - 创建必要的目录结构和初始化文件
"""

import os
from pathlib import Path

def create_directory_structure():
    """创建项目目录结构"""
    print("创建项目目录结构...")
    
    directories = [
        'agent',
        'bss', 
        'dataprocess',
        'environment',
        'models',
        'optimization',
        'scheduler',
        'simulation', 
        'utils',
        'data',
        'results',
        'configs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        
        # 创建__init__.py文件
        init_file = Path(directory) / '__init__.py'
        if not init_file.exists():
            init_file.touch()
        
        print(f"✓ {directory}/")
    
    print("目录结构创建完成！")

def create_sample_config():
    """创建示例配置文件"""
    print("创建示例配置文件...")
    
    config = {
        "m_areas": 20,
        "L_energy_levels": 10,
        "T_periods": 96,
        "period_length_minutes": 15,
        "num_taxis": 500,
        "num_stations": 10,
        "station_capacity": 30,
        "station_chargers": 8,
        "beta": -0.1,
        "solver_method": "auto",
        "time_limit": 300,
        "mip_gap": 0.05,
        "data_filepath": "data/yellow_tripdata_2025-01.parquet",
        "use_sample": True,
        "sample_size": 10000,
        "simulation_duration": 360,
        "start_hour": 6,
        "random_seed": 42,
        "output_dir": "results",
        "save_detailed_results": True
    }
    
    import json
    config_file = Path('configs') / 'default_config.json'
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ 创建配置文件: {config_file}")

def create_sample_data():
    """创建示例数据文件"""
    print("创建示例数据文件...")
    
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # 创建示例NYC出租车数据
        np.random.seed(42)
        n_rows = 5000
        
        # 生成时间戳
        start_time = pd.Timestamp('2025-01-01 06:00:00')
        pickup_times = [start_time + pd.Timedelta(minutes=int(x)) 
                       for x in np.random.uniform(0, 1440, n_rows)]
        
        # 生成NYC坐标范围内的经纬度
        pickup_lat = np.random.uniform(40.7, 40.8, n_rows)
        pickup_lon = np.random.uniform(-74.02, -73.95, n_rows)
        dropoff_lat = np.random.uniform(40.7, 40.8, n_rows)
        dropoff_lon = np.random.uniform(-74.02, -73.95, n_rows)
        
        # 创建数据
        data = {
            'tpep_pickup_datetime': pickup_times,
            'pickup_latitude': pickup_lat,
            'pickup_longitude': pickup_lon,
            'dropoff_latitude': dropoff_lat,
            'dropoff_longitude': dropoff_lon,
            'passenger_count': np.random.randint(1, 5, n_rows),
            'trip_distance': np.random.uniform(0.5, 10, n_rows),
            'fare_amount': np.random.uniform(5, 50, n_rows),
            'taxi_id': np.random.randint(1, 501, n_rows)
        }
        
        df = pd.DataFrame(data)
        
        # 保存为parquet格式
        data_file = Path('data') / 'yellow_tripdata_2025-01.parquet'
        df.to_parquet(data_file, index=False)
        
        print(f"✓ 创建示例数据文件: {data_file} ({len(df)} 条记录)")
        
    except ImportError:
        print("⚠ 无法创建示例数据文件，请确保安装了pandas和pyarrow")
    except Exception as e:
        print(f"⚠ 创建示例数据时出错: {e}")

def create_readme():
    """创建快速开始说明"""
    print("创建快速开始说明...")
    
    readme_content = """# 快速开始指南

## 1. 环境设置

```bash
# 激活conda环境
conda activate bsm

# 或者安装依赖
pip install -r requirements.txt
```

## 2. 运行测试

```bash
# 测试导入和基本功能
python test_imports.py

# 运行基本模拟
python main.py --help
```

## 3. 基本使用

```bash
# 使用默认配置运行模拟
python main.py

# 使用示例配置
python main.py --config configs/default_config.json

# 自定义参数
python main.py --duration 480 --stations 15 --areas 25
```

## 4. 配置文件

配置文件位于 `configs/default_config.json`，包含所有可调参数：

- `m_areas`: 城市区域数量
- `L_energy_levels`: 电池能量等级数量  
- `T_periods`: 时间段数量
- `num_taxis`: 出租车数量
- `num_stations`: 换电站数量
- `solver_method`: 优化方法 ('gurobi', 'heuristic', 'auto')

## 5. 数据文件

- 示例数据: `data/yellow_tripdata_2025-01.parquet`
- 真实NYC数据: 下载后放置在 `data/` 目录

## 6. 结果输出

模拟结果保存在 `results_YYYYMMDD_HHMMSS/` 目录：

- `simulation_results.json`: 完整结果数据
- `performance_report.txt`: 性能报告
- `*.png`: 可视化图表

## 7. 论文对应关系

| 论文章节 | 实现文件 |
|---------|---------|
| III-A E-taxi Systems | models/state_transition.py |
| III-B BSS Model | models/state_transition.py |
| III-C Fleet Optimization | models/optimization_model.py |
| 公式(1)-(7) | ETaxiStateModel |
| 公式(11)-(12) | JointOptimizationModel |

## 故障排除

如果遇到导入错误，请运行：
```bash
python setup_project.py
python test_imports.py
```
"""
    
    with open('QUICKSTART.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("✓ 创建快速开始指南: QUICKSTART.md")

def create_requirements():
    """创建requirements.txt"""
    print("创建requirements.txt...")
    
    requirements = """# 基础依赖
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0

# 数据处理
pyarrow>=5.0.0
openpyxl>=3.0.0

# 机器学习 (可选)
scikit-learn>=1.0.0

# 地理数据 (可选)
geopandas>=0.10.0
folium>=0.12.0

# 优化器 (可选)
# gurobipy>=9.5.0  # 需要许可证

# 可视化
plotly>=5.0.0
tqdm>=4.62.0

# 其他工具
python-dateutil>=2.8.0
pytz>=2021.1
"""
    
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write(requirements)
    
    print("✓ 创建requirements.txt")

def main():
    """主设置函数"""
    print("电动出租车电池交换系统 - 项目设置")
    print("=" * 50)
    
    # 创建目录结构
    create_directory_structure()
    print()
    
    # 创建示例配置
    create_sample_config()
    print()
    
    # 创建示例数据
    create_sample_data()
    print()
    
    # 创建说明文件
    create_readme()
    print()
    
    # 创建requirements.txt
    create_requirements()
    print()
    
    print("=" * 50)
    print("项目设置完成！")
    print()
    print("下一步:")
    print("1. 运行: python test_imports.py")
    print("2. 运行: python main.py --help")
    print("3. 查看: QUICKSTART.md")
    print()
    print("如需使用真实NYC数据，请将数据文件放置在data/目录下")

if __name__ == "__main__":
    main()