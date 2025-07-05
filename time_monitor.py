#!/usr/bin/env python3
"""
实时监控Optuna超参数优化进度
支持表格显示、实时图表、状态追踪
"""

import os
import sys
import time
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.layout import Layout
from rich.text import Text
import threading
import argparse

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

class OptunaMonitor:
    def __init__(self, db_path, study_name, refresh_interval=10):
        """
        初始化监控器
        
        Args:
            db_path: 数据库文件路径
            study_name: Study名称
            refresh_interval: 刷新间隔（秒）
        """
        self.db_path = db_path
        self.study_name = study_name
        self.refresh_interval = refresh_interval
        self.console = Console()
        self.running = True
        
        # 数据存储
        self.trial_history = []
        self.current_trial = None
        self.best_trial = None
        
        # 图表相关
        self.fig = None
        self.axes = None
        
    def connect_database(self):
        """连接数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            return conn
        except Exception as e:
            self.console.print(f"[red]数据库连接失败: {e}[/red]")
            return None
    
    def get_trial_data(self):
        """获取trial数据"""
        conn = self.connect_database()
        if not conn:
            return pd.DataFrame()
        
        try:
            # 查询trials数据
            query = """
            SELECT 
                trial_id,
                number,
                value,
                state,
                datetime_start,
                datetime_complete,
                params_json
            FROM trials 
            WHERE study_id = (
                SELECT study_id FROM studies WHERE study_name = ?
            )
            ORDER BY number
            """
            
            df = pd.read_sql_query(query, conn, params=[self.study_name])
            conn.close()
            
            if not df.empty:
                # 解析参数
                import json
                df['params'] = df['params_json'].apply(
                    lambda x: json.loads(x) if x else {}
                )
                
                # 处理时间
                df['datetime_start'] = pd.to_datetime(df['datetime_start'])
                df['datetime_complete'] = pd.to_datetime(df['datetime_complete'])
                df['duration'] = (df['datetime_complete'] - df['datetime_start']).dt.total_seconds() / 60
                
            return df
            
        except Exception as e:
            self.console.print(f"[red]查询数据失败: {e}[/red]")
            conn.close()
            return pd.DataFrame()
    
    def create_status_table(self, df):
        """创建状态表格"""
        table = Table(title="🚀 Optuna超参数优化实时监控")
        
        # 添加列
        table.add_column("Trial", style="cyan", width=6)
        table.add_column("状态", style="magenta", width=8)
        table.add_column("MAE", style="green", width=10)
        table.add_column("学习率", style="yellow", width=10)
        table.add_column("Dropout", style="blue", width=8)
        table.add_column("Hidden", style="red", width=8)
        table.add_column("Batch", style="purple", width=8)
        table.add_column("耗时(分钟)", style="white", width=10)
        
        if df.empty:
            table.add_row("--", "等待中", "--", "--", "--", "--", "--", "--")
            return table
        
        # 获取最近10个trials
        recent_trials = df.tail(10)
        
        for _, trial in recent_trials.iterrows():
            # 状态处理
            if trial['state'] == 'COMPLETE':
                status = "✅ 完成"
                status_style = "green"
            elif trial['state'] == 'PRUNED':
                status = "✂️ 剪枝"
                status_style = "yellow"
            elif trial['state'] == 'RUNNING':
                status = "🏃 运行中"
                status_style = "blue"
            elif trial['state'] == 'FAIL':
                status = "❌ 失败"
                status_style = "red"
            else:
                status = "⏸️ 等待"
                status_style = "white"
            
            # 参数提取
            params = trial['params']
            lr = f"{params.get('lr', 0):.4f}" if 'lr' in params else "--"
            dropout = f"{params.get('dropout_rate', 0):.3f}" if 'dropout_rate' in params else "--"
            hidden = str(params.get('hidden_features', '--'))
            batch = str(params.get('batch_size', '--'))
            
            # MAE值
            mae = f"{trial['value']:.6f}" if pd.notna(trial['value']) else "--"
            
            # 耗时
            duration = f"{trial['duration']:.1f}" if pd.notna(trial['duration']) else "--"
            
            table.add_row(
                str(trial['number']),
                f"[{status_style}]{status}[/{status_style}]",
                mae,
                lr,
                dropout,
                hidden,
                batch,
                duration
            )
        
        return table
    
    def create_summary_panel(self, df):
        """创建摘要面板"""
        if df.empty:
            summary_text = """
🔍 [bold blue]优化状态[/bold blue]
   总试验数: 0
   完成试验: 0
   最佳MAE: --
   平均耗时: --
   
🎯 [bold green]当前最佳参数[/bold green]
   等待数据...
            """
        else:
            total_trials = len(df)
            completed_trials = len(df[df['state'] == 'COMPLETE'])
            best_mae = df[df['state'] == 'COMPLETE']['value'].min() if completed_trials > 0 else None
            avg_duration = df[df['state'] == 'COMPLETE']['duration'].mean() if completed_trials > 0 else None
            
            # 获取最佳trial参数
            if best_mae is not None:
                best_trial = df[df['value'] == best_mae].iloc[0]
                best_params = best_trial['params']
                
                best_params_text = f"""
   Trial #{best_trial['number']}
   学习率: {best_params.get('lr', '--'):.4f}
   Dropout: {best_params.get('dropout_rate', '--'):.3f}
   Hidden: {best_params.get('hidden_features', '--')}
   Batch: {best_params.get('batch_size', '--')}"""
            else:
                best_params_text = "   等待完成试验..."
            
            summary_text = f"""
🔍 [bold blue]优化状态[/bold blue]
   总试验数: {total_trials}
   完成试验: {completed_trials}
   最佳MAE: {best_mae:.6f if best_mae else '--'}
   平均耗时: {avg_duration:.1f}分钟 if avg_duration else '--'
   
🎯 [bold green]当前最佳参数[/bold green]{best_params_text}
            """
        
        return Panel(summary_text, title="📊 实时统计", border_style="green")
    
    def save_real_time_plots(self, df, output_dir="./monitoring_plots"):
        """保存实时图表"""
        if df.empty:
            return
        
        os.makedirs(output_dir, exist_ok=True)
        completed_df = df[df['state'] == 'COMPLETE'].copy()
        
        if completed_df.empty:
            return
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Optuna超参数优化实时监控', fontsize=16, fontweight='bold')
        
        # 1. 优化历史
        axes[0, 0].plot(completed_df['number'], completed_df['value'], 'o-', linewidth=2, markersize=6)
        axes[0, 0].axhline(y=completed_df['value'].min(), color='red', linestyle='--', alpha=0.7, label=f'最佳: {completed_df["value"].min():.6f}')
        axes[0, 0].set_xlabel('Trial Number')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].set_title('优化历史')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 参数分布 - 学习率 vs MAE
        if 'lr' in completed_df['params'].iloc[0]:
            lrs = [p.get('lr', np.nan) for p in completed_df['params']]
            scatter = axes[0, 1].scatter(lrs, completed_df['value'], c=completed_df['number'], cmap='viridis', s=60, alpha=0.7)
            axes[0, 1].set_xlabel('Learning Rate')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].set_title('学习率 vs MAE')
            axes[0, 1].set_xscale('log')
            plt.colorbar(scatter, ax=axes[0, 1], label='Trial Number')
        
        # 3. 训练时长分布
        axes[1, 0].hist(completed_df['duration'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].axvline(x=completed_df['duration'].mean(), color='red', linestyle='--', label=f'平均: {completed_df["duration"].mean():.1f}分钟')
        axes[1, 0].set_xlabel('训练时长 (分钟)')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].set_title('训练时长分布')
        axes[1, 0].legend()
        
        # 4. 当前状态饼图
        state_counts = df['state'].value_counts()
        colors = {'COMPLETE': '#2ecc71', 'RUNNING': '#3498db', 'PRUNED': '#f39c12', 'FAIL': '#e74c3c'}
        pie_colors = [colors.get(state, '#95a5a6') for state in state_counts.index]
        
        axes[1, 1].pie(state_counts.values, labels=state_counts.index, autopct='%1.1f%%', 
                      colors=pie_colors, startangle=90)
        axes[1, 1].set_title('试验状态分布')
        
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = f"{output_dir}/optimization_monitor_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def run_terminal_monitor(self):
        """运行终端监控"""
        with Live(refresh_per_second=1) as live:
            while self.running:
                try:
                    # 获取数据
                    df = self.get_trial_data()
                    
                    # 创建布局
                    layout = Layout()
                    layout.split_column(
                        Layout(name="summary", size=10),
                        Layout(name="table")
                    )
                    
                    # 更新内容
                    layout["summary"].update(self.create_summary_panel(df))
                    layout["table"].update(self.create_status_table(df))
                    
                    # 更新显示
                    live.update(layout)
                    
                    # 保存图表（每分钟一次）
                    if int(time.time()) % 60 == 0:
                        self.save_real_time_plots(df)
                    
                    time.sleep(self.refresh_interval)
                    
                except KeyboardInterrupt:
                    self.console.print("\n[yellow]监控已停止[/yellow]")
                    self.running = False
                    break
                except Exception as e:
                    self.console.print(f"[red]监控错误: {e}[/red]")
                    time.sleep(5)
    
    def generate_final_report(self, output_dir="./final_report"):
        """生成最终报告（用于论文）"""
        df = self.get_trial_data()
        if df.empty:
            self.console.print("[red]没有数据可生成报告[/red]")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        completed_df = df[df['state'] == 'COMPLETE'].copy()
        
        if completed_df.empty:
            self.console.print("[red]没有完成的试验，无法生成报告[/red]")
            return
        
        # 设置论文级别的图表样式
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
        
        # 创建多个图表
        
        # 1. 优化收敛图（论文标准）
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # 计算累计最佳值
        cumulative_best = completed_df['value'].cummin()
        
        ax.plot(completed_df['number'], completed_df['value'], 'o', alpha=0.6, label='Trial Results', markersize=4)
        ax.plot(completed_df['number'], cumulative_best, 'r-', linewidth=2, label='Best So Far')
        ax.fill_between(completed_df['number'], cumulative_best, alpha=0.2, color='red')
        
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Mean Absolute Error (MAE)')
        ax.set_title('Hyperparameter Optimization Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        best_mae = completed_df['value'].min()
        improvement = ((completed_df['value'].iloc[0] - best_mae) / completed_df['value'].iloc[0]) * 100
        ax.text(0.02, 0.98, f'Best MAE: {best_mae:.6f}\nImprovement: {improvement:.1f}%', 
                transform=ax.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/optimization_convergence.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_dir}/optimization_convergence.pdf", bbox_inches='tight')  # 论文用PDF
        plt.close()
        
        # 2. 参数重要性热力图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 提取参数
        param_names = ['lr', 'dropout_rate', 'hidden_features', 'batch_size']
        param_data = {}
        
        for param in param_names:
            param_data[param] = [p.get(param, np.nan) for p in completed_df['params']]
        
        param_df = pd.DataFrame(param_data)
        param_df['mae'] = completed_df['value'].values
        
        # 学习率 vs MAE
        axes[0, 0].scatter(param_df['lr'], param_df['mae'], alpha=0.7, s=60)
        axes[0, 0].set_xlabel('Learning Rate')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].set_title('Learning Rate vs MAE')
        axes[0, 0].set_xscale('log')
        
        # Dropout vs MAE  
        axes[0, 1].scatter(param_df['dropout_rate'], param_df['mae'], alpha=0.7, s=60, color='orange')
        axes[0, 1].set_xlabel('Dropout Rate')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_title('Dropout Rate vs MAE')
        
        # Hidden Features vs MAE
        axes[1, 0].boxplot([param_df[param_df['hidden_features'] == hf]['mae'].dropna() 
                           for hf in sorted(param_df['hidden_features'].dropna().unique())],
                          labels=sorted(param_df['hidden_features'].dropna().unique()))
        axes[1, 0].set_xlabel('Hidden Features')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].set_title('Hidden Features vs MAE')
        
        # Batch Size vs MAE
        axes[1, 1].boxplot([param_df[param_df['batch_size'] == bs]['mae'].dropna() 
                           for bs in sorted(param_df['batch_size'].dropna().unique())],
                          labels=sorted(param_df['batch_size'].dropna().unique()))
        axes[1, 1].set_xlabel('Batch Size')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].set_title('Batch Size vs MAE')
        
        plt.suptitle('Hyperparameter Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/parameter_analysis.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_dir}/parameter_analysis.pdf", bbox_inches='tight')
        plt.close()
        
        # 3. 生成统计报告
        report = f"""
# Hyperparameter Optimization Report

## Summary Statistics
- Total Trials: {len(df)}
- Completed Trials: {len(completed_df)}
- Success Rate: {len(completed_df)/len(df)*100:.1f}%
- Best MAE: {best_mae:.6f}
- Average Trial Duration: {completed_df['duration'].mean():.1f} minutes

## Best Parameters
{completed_df.loc[completed_df['value'].idxmin(), 'params']}

## Trial Statistics
{completed_df[['value', 'duration']].describe()}
        """
        
        with open(f"{output_dir}/optimization_report.md", 'w') as f:
            f.write(report)
        
        # 4. 保存CSV数据
        completed_df.to_csv(f"{output_dir}/trial_results.csv", index=False)
        
        self.console.print(f"[green]报告已生成到: {output_dir}[/green]")
        self.console.print(f"[green]包含: PNG/PDF图表, CSV数据, Markdown报告[/green]")


def find_latest_database(search_dir="./output_4090"):
    """查找最新的数据库文件"""
    db_files = []
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file.endswith('.db'):
                db_path = os.path.join(root, file)
                db_files.append((db_path, os.path.getmtime(db_path)))
    
    if not db_files:
        return None
    
    # 返回最新的数据库
    return max(db_files, key=lambda x: x[1])[0]


def main():
    parser = argparse.ArgumentParser(description="Optuna实时监控工具")
    parser.add_argument("--db_path", type=str, help="数据库文件路径")
    parser.add_argument("--study_name", type=str, default="GCPNet_4090_hyperopt", help="Study名称")
    parser.add_argument("--refresh_interval", type=int, default=10, help="刷新间隔（秒）")
    parser.add_argument("--report_only", action="store_true", help="仅生成最终报告")
    parser.add_argument("--search_dir", type=str, default="./output_4090", help="搜索目录")
    
    args = parser.parse_args()
    
    # 自动查找数据库
    if not args.db_path:
        args.db_path = find_latest_database(args.search_dir)
        if not args.db_path:
            print("❌ 没有找到数据库文件")
            return
        print(f"📁 使用数据库: {args.db_path}")
    
    if not os.path.exists(args.db_path):
        print(f"❌ 数据库文件不存在: {args.db_path}")
        return
    
    monitor = OptunaMonitor(args.db_path, args.study_name, args.refresh_interval)
    
    if args.report_only:
        # 仅生成报告
        monitor.generate_final_report()
    else:
        # 启动实时监控
        try:
            monitor.run_terminal_monitor()
        except KeyboardInterrupt:
            print("\n监控已停止")
        finally:
            # 生成最终报告
            monitor.generate_final_report()


if __name__ == "__main__":
    main()