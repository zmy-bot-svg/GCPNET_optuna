# ===================================================================
# 最终修正版 JARVIS 数据处理流水线 (V3.1)
# 修正: 修复了 _load_cache 函数中的 NameError (变量'f'未定义) 问题。
# ===================================================================

# 确保已安装必要的库
# pip install -U jarvis-tools pandas matplotlib seaborn numpy

import os
import pickle
from datetime import datetime
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jarvis.db.figshare import data

# --- 全局设置 ---
warnings.filterwarnings('ignore')
# 在您的环境中可能需要安装中文字体，或替换为 'Microsoft YaHei' 等
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("中文字体'SimHei'未找到，绘图标题可能显示异常。")


class JarvisConfig:
    """配置管理类：集中管理不同的性质组合包，便于切换研究方向。"""
    # 最终用于MTL建模的、高度相关的“小而精”性质组合包
    ELECTRONIC_MTL_BUNDLE = [
        'optb88vdw_bandgap',          # 在 'dft_3d_2021' 中可用的带隙
        'dfpt_piezo_max_dielectric',   # 介电常数
        'dft_3d_avg_elec_mass',        # 电子有效质量
        'dft_3d_avg_hole_mass'         # 空穴有效质量
    ]


class JarvisDataHandler:
    """
    (V3.1-健壮版) 一个稳健的JARVIS数据处理器，辅助研究者进行数据探索和准备。
    """
    
    def __init__(self, cache_dir='jarvis_cache'):
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir
        self.raw_data_cache_file = os.path.join(cache_dir, 'dft_3d_2021_full.pkl')

    def _save_cache(self, df):
        """保存完整数据集的缓存"""
        try:
            with open(self.raw_data_cache_file, 'wb') as f:
                pickle.dump(df, f)
            print(f"原始数据已成功缓存到: {self.raw_data_cache_file}")
        except Exception as e:
            print(f"保存缓存失败: {e}")

    # =========================================================================
    # !! 关键修正部分 !!
    # =========================================================================
    def _load_cache(self):
        """(经修正) 从本地文件加载缓存数据"""
        if os.path.exists(self.raw_data_cache_file):
            print(f"从缓存文件 {self.raw_data_cache_file} 加载...")
            try:
                # 确保 'f' 在 with 语句中被定义和使用，并把加载的数据赋给一个外部变量
                with open(self.raw_data_cache_file, 'rb') as f:
                    loaded_data = pickle.load(f)
                return loaded_data
            except Exception as e:
                # 如果加载失败（例如，文件损坏），打印错误并返回None
                print(f"加载缓存失败: {e}")
                return None
        # 如果文件不存在，也返回None
        return None
    # =========================================================================

    def download_jarvis_data(self, use_cache=True, max_retries=3):
        """(经加固) 下载或从缓存加载完整的dft_3d_2021数据集，包含重试机制。"""
        if use_cache:
            df = self._load_cache()
            if df is not None and not df.empty:
                print("✓ 缓存加载成功。")
                return df
            elif df is not None and df.empty:
                 print("警告：缓存数据为空，将尝试重新下载...")

        for attempt in range(max_retries):
            try:
                print(f"正在从 JARVIS 数据库下载 'dft_3d_2021' ... [尝试 {attempt + 1}/{max_retries}]")
                df = pd.DataFrame(data('dft_3d_2021'))
                if df.empty: raise ValueError("下载的数据集为空")
                print(f"✓ 下载成功！共获取 {len(df)} 条材料记录。")
                self._save_cache(df)
                return df
            except Exception as e:
                print(f"✗ 第 {attempt + 1} 次尝试失败: {e}")
                if attempt == max_retries - 1:
                    print("已达到最大尝试次数，下载失败。")
        return None

    def explore_completeness(self, df):
        """(探索功能) 生成并显示数据完整性报告，辅助决策。"""
        print("\n" + "="*60)
        print(">> 步骤 1: 探索数据完整性 <<")
        print("="*60)
        
        report = [{'property': col, 'valid_ratio(%)': df[col].notna().mean() * 100} 
                  for col in df.columns if col not in ['jid', 'atoms', 'atoms_obj']]
        
        report_df = pd.DataFrame(report).sort_values(by='valid_ratio(%)', ascending=False)
        report_file = os.path.join(self.cache_dir, 'data_completeness_report.csv')
        report_df.to_csv(report_file, index=False, float_format='%.2f')
        
        print(f"数据完整性报告已生成 (详情请见: {report_file})")
        print("以下是数据最完整的部分性质：")
        print(report_df.head(15).to_string())
        return report_df

    def plot_correlation_for_bundle(self, df, property_bundle):
        """(探索功能) 为指定的性质组合绘制相关性矩阵。"""
        available_bundle = [p for p in property_bundle if p in df.columns]
        if len(available_bundle) < 2: 
            print(f"警告: 在数据中找到的目标性质少于2个 ({available_bundle})，无法绘制相关性矩阵。")
            return

        print(f"\n" + "="*60)
        print(f">> 步骤 2: 探索性质相关性 <<")
        print(f"为以下 {len(available_bundle)} 个可用性质绘制相关性矩阵: {available_bundle}")
        print("="*60)
        
        corr_data = df[available_bundle].corr()
        
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_data, dtype=bool))
        sns.heatmap(corr_data, mask=mask, annot=True, cmap='RdBu_r', fmt='.2f', square=True)
        plt.title('性质相关性矩阵', fontsize=16)
        plt.show()

    def prepare_final_dataset(self, df, property_bundle):
        """(准备功能) 根据最终确定的性质组合，准备用于建模的干净数据集。"""
        print("\n" + "="*60)
        print(">> 步骤 3: 准备最终建模数据集 <<")
        print("="*60)

        available_bundle = [p for p in property_bundle if p in df.columns]
        missing_props = [p for p in property_bundle if p not in df.columns]
        
        if missing_props:
            print(f"提示：以下目标性质在数据集中不存在，已被自动忽略：\n{missing_props}")
            
        if not available_bundle:
            print("错误: 所有目标性质均不在数据集中！")
            return pd.DataFrame()

        print(f"将基于以下实际存在的性质构建MTL数据集:\n{available_bundle}")
        
        cols_to_get = ['jid', 'atoms'] + available_bundle
        final_df = df[cols_to_get].dropna(subset=available_bundle).reset_index(drop=True)
        
        print(f"清洗后，获得 {len(final_df)} 个包含所有目标性质的样本。")
        return final_df

def main():
    """主函数 - 演示一个清晰、由研究者主导的科研流程"""
    config = JarvisConfig()
    handler = JarvisDataHandler()
    
    # 获取原始数据
    raw_df = handler.download_jarvis_data()
    if raw_df is None: return

    # --- 步骤 1: 探索 ---
    handler.explore_completeness(raw_df)

    # --- 步骤 2: 决策与分析 ---
    # 研究者根据探索报告和研究目标，决定最终的性质组合
    FINAL_MTL_BUNDLE = config.ELECTRONIC_MTL_BUNDLE
    handler.plot_correlation_for_bundle(raw_df, FINAL_MTL_BUNDLE)
    
    # --- 步骤 3: 准备 ---
    final_dataset = handler.prepare_final_dataset(raw_df, FINAL_MTL_BUNDLE)
    
    if not final_dataset.empty:
        print("\n最终数据集准备就绪，可用于模型训练:")
        print(final_dataset.head())

if __name__ == "__main__":
    main()