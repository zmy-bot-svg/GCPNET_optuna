# 监控脚本 - 请从artifacts中复制完整内容
#!/usr/bin/env python3
"""
GCPNet 实时训练监控脚本
使用方法：
1. 在训练开始前运行：python monitor.py &
2. 查看实时日志：tail -f monitor.log
3. 停止监控：kill %1 或者找到PID后 kill PID
"""

import os
import time
import json
import sqlite3
import psutil
import torch
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import threading
import signal
import sys
import argparse

class TrainingMonitor:
    def __init__(self, check_interval=30, log_file="monitor.log"):
        self.check_interval = check_interval
        self.log_file = log_file
        self.running = True
        
        # 自动查找数据库文件
        self.db_path = self._find_database()
        
        # 初始化监控数据
        self.last_completed_trial = -1
        self.gpu_memory_history = []
        self.stuck_trial_threshold = 600  # 10分钟没有进展认为卡住
        
        print(f"🚀 训练监控器启动")
        print(f"📊 监控数据库: {self.db_path if self.db_path else '未找到'}")
        print(f"⏱️  检查间隔: {self.check_interval}秒")
        print(f"📝 日志文件: {self.log_file}")
        print("="*60)

    def _find_database(self):
        """自动查找Optuna数据库文件"""
        # 在当前目录及子目录下查找.db文件
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith(".db") and ("optuna" in file.lower() or "hyperopt" in file.lower()):
                    found_path = os.path.join(root, file)
                    print(f"🔍 自动发现数据库: {found_path}")
                    return found_path
        
        print("⚠️  未找到Optuna数据库文件")
        return None

    def log(self, message, level="INFO"):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}][{level}] {message}"
        print(log_message)
        
        # 写入日志文件
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")

    def get_gpu_memory_info(self):
        """获取GPU内存信息"""
        if not torch.cuda.is_available():
            return None
        
        try:
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            free = total - cached
            
            return {
                "total": total,
                "allocated": allocated,
                "cached": cached,
                "free": free,
                "utilization": (allocated / total) * 100 if total > 0 else 0
            }
        except Exception as e:
            self.log(f"获取GPU信息失败: {e}", "ERROR")
            return None

    def get_optuna_status(self):
        """获取Optuna试验状态"""
        if not self.db_path or not os.path.exists(self.db_path):
            return None
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 获取试验统计
            cursor.execute("""
                SELECT state, COUNT(*) 
                FROM trials 
                GROUP BY state
            """)
            state_counts = dict(cursor.fetchall())
            
            # 获取最新的试验信息
            cursor.execute("""
                SELECT trial_id, state, datetime_start, datetime_complete, value
                FROM trials 
                ORDER BY trial_id DESC 
                LIMIT 5
            """)
            recent_trials = cursor.fetchall()
            
            # 获取最佳试验
            cursor.execute("""
                SELECT trial_id, value, datetime_complete
                FROM trials 
                WHERE state = 'COMPLETE' AND value IS NOT NULL
                ORDER BY value ASC 
                LIMIT 1
            """)
            best_trial = cursor.fetchone()
            
            conn.close()
            
            return {
                "state_counts": state_counts,
                "recent_trials": recent_trials,
                "best_trial": best_trial
            }
            
        except Exception as e:
            self.log(f"读取Optuna数据库失败: {e}", "ERROR")
            return None

    def get_training_processes(self):
        """获取训练相关的Python进程"""
        training_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'cpu_percent', 'create_time']):
            try:
                if proc.info['name'] == 'python' and proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline'])
                    if any(keyword in cmdline.lower() for keyword in ['main.py', 'optuna', 'hyperparameter']):
                        training_processes.append({
                            'pid': proc.info['pid'],
                            'cmdline': cmdline,
                            'memory_mb': proc.info['memory_info'].rss / 1024 / 1024,
                            'cpu_percent': proc.info['cpu_percent'],
                            'running_time': time.time() - proc.info['create_time']
                        })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return training_processes

    def detect_stuck_trial(self, optuna_status):
        """检测卡住的试验"""
        if not optuna_status or not optuna_status['recent_trials']:
            return None
        
        # 获取最新的RUNNING试验
        running_trials = [t for t in optuna_status['recent_trials'] if t[1] == 'RUNNING']
        
        if not running_trials:
            return None
        
        latest_running = running_trials[0]
        trial_id = latest_running[0]
        start_time_str = latest_running[2]
        
        if start_time_str:
            try:
                # 解析时间字符串
                start_time = datetime.fromisoformat(start_time_str.replace('Z', ''))
                current_time = datetime.now()
                elapsed = (current_time - start_time).total_seconds()
                
                if elapsed > self.stuck_trial_threshold:
                    return {
                        "trial_id": trial_id,
                        "elapsed_seconds": elapsed,
                        "start_time": start_time_str
                    }
            except Exception as e:
                self.log(f"解析试验时间失败: {e}", "WARNING")
        
        return None

    def check_memory_leak(self, gpu_info):
        """检测GPU内存泄漏"""
        if not gpu_info:
            return False
        
        self.gpu_memory_history.append({
            "timestamp": datetime.now(),
            "allocated": gpu_info["allocated"],
            "cached": gpu_info["cached"]
        })
        
        # 只保留最近30分钟的数据
        cutoff = datetime.now() - timedelta(minutes=30)
        self.gpu_memory_history = [
            h for h in self.gpu_memory_history 
            if h["timestamp"] > cutoff
        ]
        
        # 检测内存是否持续增长
        if len(self.gpu_memory_history) >= 6:  # 至少3分钟的数据
            recent = self.gpu_memory_history[-3:]
            older = self.gpu_memory_history[-6:-3]
            
            recent_avg = sum(h["allocated"] for h in recent) / len(recent)
            older_avg = sum(h["allocated"] for h in older) / len(older)
            
            # 如果内存使用增长超过0.5GB，认为可能有泄漏
            if recent_avg - older_avg > 0.5:
                return True
        
        return False

    def run_check(self):
        """执行一次完整检查"""
        # 获取各种状态信息
        gpu_info = self.get_gpu_memory_info()
        optuna_status = self.get_optuna_status()
        training_processes = self.get_training_processes()
        
        # 检测问题
        stuck_trial = self.detect_stuck_trial(optuna_status)
        memory_leak = self.check_memory_leak(gpu_info)
        
        # 生成报告
        report = self.generate_report(gpu_info, optuna_status, training_processes, stuck_trial, memory_leak)
        self.log(report, "REPORT")
        
        # 检查重要事件
        if stuck_trial:
            self.log(f"🚨 检测到卡住的试验: Trial #{stuck_trial['trial_id']} (已运行{stuck_trial['elapsed_seconds']/60:.1f}分钟)", "WARNING")
        
        if memory_leak:
            self.log("🚨 检测到可能的GPU内存泄漏", "WARNING")
        
        # 检查是否有新完成的trial
        if optuna_status and 'COMPLETE' in optuna_status['state_counts']:
            completed_count = optuna_status['state_counts']['COMPLETE']
            if completed_count > self.last_completed_trial:
                self.log(f"✅ 新完成的试验，总计: {completed_count}个", "INFO")
                self.last_completed_trial = completed_count

    def generate_report(self, gpu_info, optuna_status, training_processes, stuck_trial, memory_leak):
        """生成监控报告"""
        report_lines = []
        
        # GPU状态
        if gpu_info:
            report_lines.append(f"🎮 GPU - 已用:{gpu_info['allocated']:.1f}GB({gpu_info['utilization']:.1f}%) 可用:{gpu_info['free']:.1f}GB")
            if gpu_info['free'] < 2.0:
                report_lines.append("⚠️  GPU可用内存不足2GB")
            if memory_leak:
                report_lines.append("🚨 检测到GPU内存泄漏趋势")
        else:
            report_lines.append("❌ GPU不可用")
        
        # Optuna状态
        if optuna_status:
            state_counts = optuna_status['state_counts']
            total_trials = sum(state_counts.values())
            complete_count = state_counts.get('COMPLETE', 0)
            running_count = state_counts.get('RUNNING', 0)
            failed_count = state_counts.get('FAIL', 0)
            
            report_lines.append(f"📊 Trials - 总计:{total_trials} 完成:{complete_count} 运行:{running_count} 失败:{failed_count}")
            
            if optuna_status['best_trial']:
                best = optuna_status['best_trial']
                report_lines.append(f"🏆 最佳: Trial#{best[0]} MAE:{best[1]:.6f}")
            
            if stuck_trial:
                report_lines.append(f"🚨 Trial#{stuck_trial['trial_id']} 可能卡住({stuck_trial['elapsed_seconds']/60:.1f}分钟)")
        else:
            report_lines.append("❌ 无Optuna数据")
        
        # 训练进程
        if training_processes:
            for proc in training_processes[:2]:  # 只显示前2个进程
                runtime_hours = proc['running_time'] / 3600
                report_lines.append(f"🔄 PID{proc['pid']} - {proc['memory_mb']:.0f}MB CPU:{proc['cpu_percent']:.1f}% 运行:{runtime_hours:.1f}h")
        else:
            report_lines.append("❌ 无训练进程")
        
        return " | ".join(report_lines)

    def monitor_loop(self):
        """主监控循环"""
        self.log("开始监控循环")
        
        try:
            while self.running:
                self.run_check()
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            self.log("收到中断信号，停止监控", "INFO")
        except Exception as e:
            self.log(f"监控循环出错: {e}", "ERROR")
        finally:
            self.log("监控结束", "INFO")

    def start(self):
        """启动监控"""
        def signal_handler(signum, frame):
            self.log("收到停止信号", "INFO")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        self.monitor_loop()

def main():
    parser = argparse.ArgumentParser(description="GCPNet训练实时监控")
    parser.add_argument("--interval", type=int, default=30, help="检查间隔（秒），默认30秒")
    parser.add_argument("--log-file", default="monitor.log", help="日志文件名")
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(
        check_interval=args.interval,
        log_file=args.log_file
    )
    
    monitor.start()

if __name__ == "__main__":
    main()