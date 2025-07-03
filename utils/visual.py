# simple_mae_visualization.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 设置字体
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'sans-serif']

def plot_predictions(csv_file='prediction_results.csv'):
    """
    简单的预测结果可视化脚本 - 只显示整体散点图和MAE
    """
    # 读取数据
    df = pd.read_csv(csv_file)
    print(f"Reading file: {csv_file}, Shape: {df.shape}")
    
    # 获取列名
    columns = df.columns.tolist()
    print(f"Columns: {columns}")
    
    # 根据数据格式处理
    if len(columns) == 3:
        model_col, pred_col, true_col = columns[0], columns[1], columns[2]
        
        # 清理预测值（去除方括号）
        df[pred_col] = df[pred_col].astype(str).str.strip('[]').astype(float)
        
        # 获取所有预测值和真实值
        y_true = df[true_col].values
        y_pred = df[pred_col].values
        
    else:
        print("Error: Unexpected data format")
        return
    
    # 计算总体误差指标
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # 计算R²（如果可能）
    try:
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        r2_text = f"R² = {r2:.4f}"
    except:
        r2_text = "R² = N/A"
    
    # 创建单个散点图
    plt.figure(figsize=(8, 6))
    
    # 散点图
    plt.scatter(y_true, y_pred, alpha=0.6, s=30, edgecolor='white', linewidth=0.3)
    
    # 完美预测线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='y = x (Perfect)')
    
    # 设置图表
    plt.xlabel('True Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title('Prediction vs True Values', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 设置坐标轴范围
    margin = (max_val - min_val) * 0.05 if max_val != min_val else 0.1
    plt.xlim(min_val - margin, max_val + margin)
    plt.ylim(min_val - margin, max_val + margin)
    
    # 添加统计信息文本框
    stats_text = f'MAE = {mae:.4f}\nRMSE = {rmse:.4f}\n{r2_text}\nSamples = {len(y_true)}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8),
             verticalalignment='top', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('prediction_scatter.png', dpi=300, bbox_inches='tight')
    print(f"Image saved as: prediction_scatter.png")
    plt.show()
    
    # 输出结果
    print("\nOverall Prediction Performance:")
    print("=" * 40)
    print(f"Total samples:             {len(y_true)}")
    print(f"MAE (Mean Absolute Error): {mae:.6f}")
    print(f"RMSE (Root Mean Sq Error): {rmse:.6f}")

def plot_predictions_dual_task(csv_file='prediction_results.csv'):
    """
    如果您有两个任务的数据，使用这个函数创建双子图
    """
    # 读取数据
    df = pd.read_csv(csv_file)
    print(f"Reading file: {csv_file}, Shape: {df.shape}")
    
    # 检查是否有多个任务的数据
    # 如果您的数据有特定的任务标识，可以在这里修改
    
    # 假设数据分为两部分（您可以根据实际情况调整）
    mid_point = len(df) // 2
    
    # 获取列名
    columns = df.columns.tolist()
    model_col, pred_col, true_col = columns[0], columns[1], columns[2]
    
    # 清理预测值
    df[pred_col] = df[pred_col].astype(str).str.strip('[]').astype(float)
    
    # 分割数据（您可以根据实际情况修改分割逻辑）
    task1_data = df.iloc[:mid_point]
    task2_data = df.iloc[mid_point:]
    
    # 创建双子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 任务1
    y_true1 = task1_data[true_col].values
    y_pred1 = task1_data[pred_col].values
    mae1 = mean_absolute_error(y_true1, y_pred1)
    rmse1 = np.sqrt(mean_squared_error(y_true1, y_pred1))
    
    ax1.scatter(y_true1, y_pred1, alpha=0.6, s=30, edgecolor='white', linewidth=0.3)
    min_val1 = min(y_true1.min(), y_pred1.min())
    max_val1 = max(y_true1.max(), y_pred1.max())
    ax1.plot([min_val1, max_val1], [min_val1, max_val1], 'r--', alpha=0.8, linewidth=2)
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title('Task 1')
    ax1.grid(True, alpha=0.3)
    
    # 任务2
    y_true2 = task2_data[true_col].values
    y_pred2 = task2_data[pred_col].values
    mae2 = mean_absolute_error(y_true2, y_pred2)
    rmse2 = np.sqrt(mean_squared_error(y_true2, y_pred2))
    
    ax2.scatter(y_true2, y_pred2, alpha=0.6, s=30, edgecolor='white', linewidth=0.3)
    min_val2 = min(y_true2.min(), y_pred2.min())
    max_val2 = max(y_true2.max(), y_pred2.max())
    ax2.plot([min_val2, max_val2], [min_val2, max_val2], 'r--', alpha=0.8, linewidth=2)
    ax2.set_xlabel('True Values')
    ax2.set_ylabel('Predicted Values')
    ax2.set_title('Task 2')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_dual_task.png', dpi=300, bbox_inches='tight')
    print(f"Dual task image saved as: prediction_dual_task.png")
    plt.show()
    
    # 输出结果
    print("\nDual Task Performance:")
    print("=" * 40)
    print(f"Task 1 - MAE: {mae1:.6f}, RMSE: {rmse1:.6f}, Samples: {len(y_true1)}")
    print(f"Task 2 - MAE: {mae2:.6f}, RMSE: {rmse2:.6f}, Samples: {len(y_true2)}")

if __name__ == '__main__':
    # 单任务可视化
    plot_predictions()
    
    # 如果需要双任务可视化，取消下面的注释
    # plot_predictions_dual_task()
