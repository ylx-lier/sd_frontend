"""
端口清理工具 - 手动清理被占用的端口
在启动主程序前可以运行此脚本来确保端口可用
"""

import subprocess
import os
import sys
import socket
import time

def check_port(port):
    """检查端口是否被占用"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('0.0.0.0', port))
            return True  # 端口可用
    except OSError:
        return False  # 端口被占用

def force_kill_port_windows(port):
    """Windows下强制释放端口"""
    print(f"🪟 Windows系统 - 强制清理端口 {port}")
    
    try:
        # 方法1: 使用netstat查找并终止进程
        cmd = f'netstat -ano | findstr ":{port}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        
        pids = set()
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    if 'LISTENING' in line or 'ESTABLISHED' in line:
                        pid = parts[-1]
                        if pid.isdigit() and pid != '0':
                            pids.add(pid)
        
        current_pid = str(os.getpid())
        killed_count = 0
        
        for pid in pids:
            if pid != current_pid:
                try:
                    # 先尝试普通终止
                    result1 = subprocess.run(f'taskkill /PID {pid}', 
                                           shell=True, capture_output=True, timeout=5)
                    time.sleep(1)
                    
                    # 如果普通终止失败，强制终止
                    if result1.returncode != 0:
                        subprocess.run(f'taskkill /F /PID {pid}', 
                                     shell=True, capture_output=True, timeout=5)
                    
                    print(f"✅ 已终止进程 PID: {pid}")
                    killed_count += 1
                    
                except Exception as e:
                    print(f"⚠️ 终止进程 {pid} 失败: {e}")
        
        # 方法2: 清理Python相关进程
        try:
            cmd = 'wmic process where "name=\'python.exe\'" get ProcessId,CommandLine /format:csv'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if any(keyword in line.lower() for keyword in ['gradio', 'uvicorn', 'streamlit', str(port)]):
                        parts = line.split(',')
                        if len(parts) >= 3:
                            pid = parts[-1].strip()
                            if pid.isdigit() and pid != str(os.getpid()):
                                try:
                                    subprocess.run(f'taskkill /F /PID {pid}', 
                                                 shell=True, capture_output=True, timeout=5)
                                    print(f"✅ 已终止相关Python进程 PID: {pid}")
                                    killed_count += 1
                                except:
                                    pass
        
        except Exception as e:
            print(f"⚠️ 清理Python进程失败: {e}")
        
        print(f"📊 总共终止了 {killed_count} 个进程")
        
    except Exception as e:
        print(f"❌ Windows端口清理失败: {e}")

def force_kill_port_unix(port):
    """Unix系统下强制释放端口"""
    print(f"🐧 Unix系统 - 强制清理端口 {port}")
    
    try:
        result = subprocess.run([
            "lsof", "-ti", f":{port}"
        ], capture_output=True, text=True, timeout=10)
        
        killed_count = 0
        if result.stdout:
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid.isdigit():
                    try:
                        # 先尝试SIGTERM
                        subprocess.run(["kill", "-15", pid], timeout=5)
                        time.sleep(1)
                        # 再尝试SIGKILL
                        subprocess.run(["kill", "-9", pid], timeout=5)
                        print(f"✅ 已终止进程 PID: {pid}")
                        killed_count += 1
                    except:
                        pass
        
        print(f"📊 总共终止了 {killed_count} 个进程")
        
    except Exception as e:
        print(f"❌ Unix端口清理失败: {e}")

def clean_port(port):
    """清理指定端口"""
    print(f"\n🔍 检查端口 {port}...")
    
    if check_port(port):
        print(f"✅ 端口 {port} 当前可用")
        return True
    
    print(f"⚠️ 端口 {port} 被占用，开始清理...")
    
    import platform
    if platform.system() == "Windows":
        force_kill_port_windows(port)
    else:
        force_kill_port_unix(port)
    
    # 等待一下再检查
    time.sleep(2)
    
    if check_port(port):
        print(f"✅ 端口 {port} 清理成功")
        return True
    else:
        print(f"❌ 端口 {port} 清理失败，可能需要手动处理")
        return False

def main():
    """主函数"""
    print("🧹 端口清理工具")
    print("=" * 50)
    
    # 常用的Gradio端口
    ports_to_clean = [7860, 7861, 7862, 7863]
    
    print("📋 将检查和清理以下端口:")
    for port in ports_to_clean:
        print(f"   • {port}")
    
    print("\n🚀 开始清理...")
    
    success_count = 0
    for port in ports_to_clean:
        if clean_port(port):
            success_count += 1
    
    print("\n" + "=" * 50)
    print(f"📊 清理结果: {success_count}/{len(ports_to_clean)} 个端口可用")
    
    if success_count == len(ports_to_clean):
        print("🎉 所有端口清理完成，可以启动主程序了！")
    else:
        print("⚠️ 部分端口清理失败，可能需要重启计算机或手动处理")
    
    input("\n按回车键退出...")

if __name__ == "__main__":
    main()
