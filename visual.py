#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optuna 超参数优化结果可视化脚本
用于生成类似论文中的超参数搜索过程图表
"""

import optuna
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
import sqlite3
from datetime import datetime

class OptunaVisualizer:
    def __init__(self, study_name_or_path):
        """
        初始化可视化器
        
        Args:
            study_name_or_path: Optuna study 名称或数据库路径
        """
        if os.path.exists(study_name_or_path):
            # 如果是文件路径，直接加载
            self.study = optuna.load_study(
                study_name=None,
                storage=f"sqlite:///{study_name_or_path}"
            )
        else:
            # 如果是 study 名称，从默认数据库加载
            self.study = optuna.load_study(study_name=study_name_or_path)
    
    def plot_optimization_history(self, save_path=None):
        """
        绘制优化历史曲线
        显示每次试验的目标值和累积最佳值
        """
        trials = self.study.trials
        trial_numbers = [trial.number for trial in trials if trial.value is not None]
        values = [trial.value for trial in trials if trial.value is not None]
        
        # 计算累积最佳值
        best_values = []
        current_best = float('inf')
        for value in values:
            if value < current_best:
                current_best = value
            best_values.append(current_best)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('每次试验的目标值', '累积最佳值趋势'),
            vertical_spacing=0.1
        )
        
        # 每次试验的值
        fig.add_trace(
            go.Scatter(
                x=trial_numbers,
                y=values,
                mode='markers+lines',
                name='Trial Value',
                marker=dict(size=6, color='lightblue'),
                line=dict(color='lightblue', width=1)
            ),
            row=1, col=1
        )
        
        # 累积最佳值
        fig.add_trace(
            go.Scatter(
                x=trial_numbers,
                y=best_values,
                mode='lines',
                name='Best Value',
                line=dict(color='red', width=3)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Optuna 超参数优化历史",
            height=600,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Trial Number", row=2, col=1)
        fig.update_yaxes(title_text="Validation MAE", row=1, col=1)
        fig.update_yaxes(title_text="Best Validation MAE", row=2, col=1)
        
        if save_path:
            fig.write_html(save_path)
            print(f"优化历史图已保存到: {save_path}")
        
        fig.show()
        return fig
    
    def plot_parameter_importance(self, save_path=None):
        """
        绘制参数重要性图
        """
        try:
            importance = optuna.importance.get_param_importances(self.study)
            
            params = list(importance.keys())
            importances = list(importance.values())
            
            fig = go.Figure(data=[
                go.Bar(
                    x=importances,
                    y=params,
                    orientation='h',
                    marker_color='lightgreen'
                )
            ])
            
            fig.update_layout(
                title="参数重要性分析",
                xaxis_title="重要性",
                yaxis_title="参数",
                height=400
            )
            
            if save_path:
                fig.write_html(save_path)
                print(f"参数重要性图已保存到: {save_path}")
            
            fig.show()
            return fig
        except Exception as e:
            print(f"无法计算参数重要性: {e}")
            return None
    
    def plot_parameter_distributions(self, save_path=None):
        """
        绘制参数分布图
        """
        trials_df = self.study.trials_dataframe()
        param_columns = [col for col in trials_df.columns if col.startswith('params_')]
        
        if not param_columns:
            print("没有找到参数数据")
            return None
        
        n_params = len(param_columns)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[col.replace('params_', '') for col in param_columns]
        )
        
        for i, param_col in enumerate(param_columns):
            row = i // n_cols + 1
            col = i % n_cols + 1
            
            values = trials_df[param_col].dropna()
            objectives = trials_df.loc[values.index, 'value']
            
            fig.add_trace(
                go.Scatter(
                    x=values,
                    y=objectives,
                    mode='markers',
                    name=param_col.replace('params_', ''),
                    marker=dict(
                        size=8,
                        color=objectives,
                        colorscale='Viridis',
                        showscale=(i == 0)  # 只在第一个子图显示色条
                    )
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="参数值与目标值的关系",
            height=200 * n_rows,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"参数分布图已保存到: {save_path}")
        
        fig.show()
        return fig
    
    def plot_parallel_coordinate(self, save_path=None):
        """
        绘制平行坐标图
        """
        trials_df = self.study.trials_dataframe()
        param_columns = [col for col in trials_df.columns if col.startswith('params_')]
        
        if not param_columns:
            print("没有找到参数数据")
            return None
        
        # 准备数据
        data = trials_df[param_columns + ['value']].dropna()
        
        # 创建平行坐标图
        dimensions = []
        for col in param_columns:
            dimensions.append(dict(
                label=col.replace('params_', ''),
                values=data[col]
            ))
        
        # 添加目标值维度
        dimensions.append(dict(
            label='Validation MAE',
            values=data['value']
        ))
        
        fig = go.Figure(data=
            go.Parcoords(
                line=dict(color=data['value'],
                         colorscale='Viridis',
                         showscale=True,
                         cmin=data['value'].min(),
                         cmax=data['value'].max()),
                dimensions=dimensions
            )
        )
        
        fig.update_layout(
            title="参数组合的平行坐标图",
            height=500
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"平行坐标图已保存到: {save_path}")
        
        fig.show()
        return fig
    
    def generate_summary_report(self, output_dir):
        """
        生成完整的可视化报告
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"正在生成可视化报告到: {output_dir}")
        
        # 生成各种图表
        self.plot_optimization_history(
            os.path.join(output_dir, "optimization_history.html")
        )
        
        self.plot_parameter_importance(
            os.path.join(output_dir, "parameter_importance.html")
        )
        
        self.plot_parameter_distributions(
            os.path.join(output_dir, "parameter_distributions.html")
        )
        
        self.plot_parallel_coordinate(
            os.path.join(output_dir, "parallel_coordinate.html")
        )
        
        # 生成文本报告
        self._generate_text_report(output_dir)
        
        print("可视化报告生成完成！")
    
    def _generate_text_report(self, output_dir):
        """
        生成文本格式的报告
        """
        report_path = os.path.join(output_dir, "optuna_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("Optuna 超参数优化报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"总试验次数: {len(self.study.trials)}\n")
            f.write(f"完成的试验: {len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])}\n")
            f.write(f"失败的试验: {len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL])}\n\n")
            
            if self.study.best_trial:
                f.write("最佳试验结果:\n")
                f.write(f"  试验编号: {self.study.best_trial.number}\n")
                f.write(f"  最佳值: {self.study.best_value:.6f}\n")
                f.write(f"  最佳参数:\n")
                for key, value in self.study.best_params.items():
                    f.write(f"    {key}: {value}\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        print(f"文本报告已保存到: {report_path}")

def main():
    """
    主函数，提供命令行接口
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Optuna 结果可视化工具")
    parser.add_argument("--study", "-s", required=True, 
                       help="Optuna study 名称或数据库文件路径")
    parser.add_argument("--output", "-o", default="./optuna_visualization",
                       help="输出目录路径")
    
    args = parser.parse_args()
    
    try:
        visualizer = OptunaVisualizer(args.study)
        visualizer.generate_summary_report(args.output)
    except Exception as e:
        print(f"可视化失败: {e}")
        print("请确保 study 名称或路径正确")

if __name__ == "__main__":
    main()