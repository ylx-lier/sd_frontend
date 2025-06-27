"""
工具函数模块 - 存储通用的辅助函数
"""

import subprocess
import os
import signal
import atexit
import sys
import socket
from datetime import datetime
import requests
from config import PROXY_CONFIG

# 全局变量用于存储Gradio应用实例和端口信息
demo_instance = None
server_port = None

def set_gradio_instance(demo, port):
    """设置Gradio实例和端口信息"""
    global demo_instance, server_port
    demo_instance = demo
    server_port = port
    print(f"📝 已记录Gradio实例和端口信息: {port}")

def get_gradio_info():
    """获取当前Gradio实例和端口信息"""
    return demo_instance, server_port

def cleanup_on_exit():
    """程序退出时的清理函数"""
    try:
        print("\n🔄 正在清理资源...")
        
        # 关闭Gradio应用
        if demo_instance is not None:
            print("📴 关闭Gradio应用...")
            try:
                # 尝试优雅关闭
                if hasattr(demo_instance, 'server') and demo_instance.server:
                    demo_instance.server.should_exit = True
                    demo_instance.server.force_exit = True
                
                # 关闭demo实例
                demo_instance.close()
                print("✅ Gradio应用已关闭")
                
                # 给一点时间让服务器完全关闭
                import time
                time.sleep(1)
                
            except Exception as e:
                print(f"⚠️ 关闭Gradio应用时出现错误: {e}")
        
        # 强制释放端口（多个端口）
        ports_to_clean = []
        if server_port is not None:
            ports_to_clean.append(server_port)
        
        # 常见的Gradio端口
        common_ports = [7860, 7861, 7862, 7863]
        for port in common_ports:
            if port not in ports_to_clean:
                ports_to_clean.append(port)
        
        for port in ports_to_clean:
            force_release_port(port)
        
        print("✅ 资源清理完成")
        
    except Exception as e:
        print(f"❌ 清理过程中出现错误: {e}")

def force_release_port(port):
    """强制释放指定端口"""
    print(f"🔓 强制释放端口 {port}...")
    
    try:
        import platform
        import subprocess
        import time
        
        if platform.system() == "Windows":
            # Windows下更强力的端口释放
            print(f"🪟 Windows系统 - 强制释放端口 {port}")
            
            # 方法1: 使用netstat和taskkill
            try:
                # 查找所有占用该端口的进程
                cmd = f'netstat -ano | findstr ":{port}"'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
                
                pids = set()
                if result.stdout:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 5:
                            # 检查是否是监听状态或已建立连接
                            if 'LISTENING' in line or 'ESTABLISHED' in line:
                                pid = parts[-1]
                                if pid.isdigit() and pid != '0':
                                    pids.add(pid)
                
                # 终止所有相关进程
                current_pid = str(os.getpid())
                for pid in pids:
                    if pid != current_pid:
                        try:
                            # 先尝试普通终止
                            result1 = subprocess.run(f'taskkill /PID {pid}', 
                                                   shell=True, capture_output=True, timeout=3)
                            time.sleep(0.5)
                            
                            # 如果普通终止失败，强制终止
                            if result1.returncode != 0:
                                subprocess.run(f'taskkill /F /PID {pid}', 
                                             shell=True, capture_output=True, timeout=3)
                            
                            print(f"🔄 已终止占用端口{port}的进程 PID: {pid}")
                        except Exception as e:
                            print(f"⚠️ 终止进程 {pid} 失败: {e}")
                
            except Exception as e:
                print(f"⚠️ netstat方法失败: {e}")
            
            # 方法2: 使用wmic查找Python/Gradio相关进程
            try:
                # 查找Python进程中可能占用端口的
                cmd = 'wmic process where "name=\'python.exe\'" get ProcessId,CommandLine /format:csv'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
                
                if result.stdout:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if 'gradio' in line.lower() or 'uvicorn' in line.lower() or str(port) in line:
                            parts = line.split(',')
                            if len(parts) >= 3:
                                pid = parts[-1].strip()
                                if pid.isdigit() and pid != str(os.getpid()):
                                    try:
                                        subprocess.run(f'taskkill /F /PID {pid}', 
                                                     shell=True, capture_output=True, timeout=3)
                                        print(f"🔄 已终止相关Python进程 PID: {pid}")
                                    except:
                                        pass
                
            except Exception as e:
                print(f"⚠️ wmic方法失败: {e}")
        
        else:
            # Unix系统下的端口释放
            print(f"🐧 Unix系统 - 强制释放端口 {port}")
            try:
                # 使用lsof查找并终止进程
                result = subprocess.run([
                    "lsof", "-ti", f":{port}"
                ], capture_output=True, text=True, timeout=5)
                
                if result.stdout:
                    pids = result.stdout.strip().split('\n')
                    for pid in pids:
                        if pid.isdigit():
                            try:
                                # 先尝试SIGTERM
                                subprocess.run(["kill", "-15", pid], timeout=3)
                                time.sleep(0.5)
                                # 再尝试SIGKILL
                                subprocess.run(["kill", "-9", pid], timeout=3)
                                print(f"🔄 已终止进程 PID: {pid}")
                            except:
                                pass
                
            except Exception as e:
                print(f"⚠️ Unix端口清理失败: {e}")
        
        # 验证端口是否已释放
        time.sleep(1)
        if is_port_available(port):
            print(f"✅ 端口 {port} 已成功释放")
        else:
            print(f"⚠️ 端口 {port} 可能仍被占用")
            
    except Exception as e:
        print(f"⚠️ 释放端口 {port} 时出现错误: {e}")

def is_port_available(port):
    """检查端口是否可用"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('0.0.0.0', port))
            return True
    except OSError:
        return False

def signal_handler(signum, frame):
    """信号处理函数"""
    signal_names = {
        signal.SIGINT: "SIGINT (Ctrl+C)",
        signal.SIGTERM: "SIGTERM (终止信号)"
    }
    
    signal_name = signal_names.get(signum, f"信号 {signum}")
    print(f"\n🛑 接收到退出信号: {signal_name}")
    print("🔄 正在优雅关闭应用...")
    
    cleanup_on_exit()
    sys.exit(0)

def setup_cleanup_handlers():
    """设置清理处理程序"""
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # 终止信号
    
    # 注册atexit处理
    atexit.register(cleanup_on_exit)
    
    print("🛡️ 已设置自动端口释放机制")

def find_free_port(start_port=7860, max_attempts=10):
    """寻找可用端口，如果端口被占用则尝试清理"""
    print(f"🔍 正在寻找可用端口（从 {start_port} 开始）...")
    
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(port):
            print(f"✅ 找到可用端口: {port}")
            return port
        else:
            print(f"⚠️ 端口 {port} 已被占用")
            
            # 尝试清理被占用的端口
            print(f"🔧 尝试清理端口 {port}...")
            force_release_port(port)
            
            # 再次检查端口是否可用
            import time
            time.sleep(1)
            if is_port_available(port):
                print(f"✅ 端口 {port} 清理成功，将使用此端口")
                return port
            else:
                print(f"❌ 端口 {port} 清理失败，继续尝试下一个端口...")
                continue
    
    print(f"❌ 在端口范围 {start_port}-{start_port + max_attempts} 内未找到可用端口")
    # 最后的努力：强制使用起始端口
    print(f"🚨 强制清理并使用端口 {start_port}")
    force_release_port(start_port)
    import time
    time.sleep(2)
    return start_port

def auto_push_to_github():
    """自动推送到 GitHub"""
    try:
        print("🚀 开始自动推送到 GitHub...")
        
        # 检查是否在 git 仓库中
        result = subprocess.run("git status", shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            return "❌ 当前目录不是 git 仓库或 git 未安装"
        
        # 添加所有更改的文件
        result = subprocess.run("git add .", shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            return f"❌ 添加文件失败: {result.stderr}"
        
        # 检查是否有更改需要提交
        result = subprocess.run("git diff --staged --quiet", shell=True, capture_output=True, text=True)
        if result.returncode == 0:  # 如果命令成功，说明没有更改
            return "✅ 没有新的更改需要提交"
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 准备提交信息
        commit_message = f"Auto update: {timestamp} - 功能更新和优化"
        
        # 提交更改
        result = subprocess.run(f'git commit -m "{commit_message}"', shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            return f"❌ 提交失败: {result.stderr}"
        
        # 推送到远程仓库
        result = subprocess.run("git push origin main", shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            return f"❌ 推送失败: {result.stderr}\n💡 请检查网络连接或 GitHub 权限"
        
        # 获取仓库 URL
        result = subprocess.run("git remote get-url origin", shell=True, capture_output=True, text=True)
        repo_url = result.stdout.strip() if result.returncode == 0 else "未知"
        
        return f"✅ 成功推送到 GitHub!\n🔗 仓库: {repo_url}\n⏰ 时间: {timestamp}"
        
    except Exception as e:
        return f"❌ 推送过程中发生错误: {str(e)}"

def test_proxy_connection(enabled, http_proxy, https_proxy):
    """测试代理连接"""
    if not enabled:
        return "❌ 代理未启用，无法测试"
    
    if not (http_proxy or https_proxy):
        return "❌ 请填写代理地址"
    
    proxies = {}
    if http_proxy:
        proxies["http"] = http_proxy
    if https_proxy:
        proxies["https"] = https_proxy
    
    try:
        # 测试连接到 Hugging Face
        response = requests.get(
            "https://huggingface.co", 
            proxies=proxies, 
            timeout=10
        )
        if response.status_code == 200:
            return "✅ 代理连接测试成功！"
        else:
            return f"⚠️ 代理连接测试失败，状态码: {response.status_code}"
    except Exception as e:
        return f"❌ 代理连接测试失败: {str(e)}"

def update_model_choices(run_mode):
    """更新模型选择器"""
    from config import get_available_models
    
    available_models = get_available_models(run_mode)
    
    if run_mode == "api":
        # API模式：只显示支持API的模型
        choices = list(available_models.keys())
        
        # 默认选择第一个推荐模型
        default_value = choices[0] if choices else "black-forest-labs/FLUX.1-dev"
        
        return {
            "choices": choices,
            "value": default_value,
            "label": "🤖 选择基础模型 (仅API支持的模型)",
            "info": "✅ API模式 - 这些模型支持云端推理，无需下载"
        }
    else:
        # 本地模式：显示所有模型
        choices = list(available_models.keys())
        return {
            "choices": choices,
            "value": "runwayml/stable-diffusion-v1-5",
            "label": "🤖 选择基础模型 (支持所有模型)",
            "info": "💾 本地模式 - 首次使用需要下载模型文件（4-10GB）"
        }
