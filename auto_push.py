#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动推送到 GitHub 的 Python 脚本
Auto Push to GitHub - Python Script
"""

import subprocess
import sys
import os
from datetime import datetime

def run_command(command, shell=True):
    """执行命令并返回结果"""
    try:
        result = subprocess.run(
            command, 
            shell=shell, 
            capture_output=True, 
            text=True, 
            encoding='utf-8'
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def auto_push_to_github():
    """自动推送到 GitHub"""
    print("🚀 开始自动推送到 GitHub...")
    
    # 检查是否在 git 仓库中
    success, _, _ = run_command("git status")
    if not success:
        print("❌ 当前目录不是 git 仓库或 git 未安装")
        return False
    
    # 添加所有更改的文件
    print("📁 添加文件到暂存区...")
    success, _, error = run_command("git add .")
    if not success:
        print(f"❌ 添加文件失败: {error}")
        return False
    
    # 检查是否有更改需要提交
    success, _, _ = run_command("git diff --staged --quiet")
    if success:  # 如果命令成功，说明没有更改
        print("✅ 没有新的更改需要提交")
        return True
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 准备提交信息
    commit_message = f"""Auto update: {timestamp}

- 代码功能更新和优化
- 添加代理支持功能
- 修复API编码问题
- 更新文档和指南
- 界面功能完善"""
    
    # 提交更改
    print("💾 提交更改...")
    success, _, error = run_command(f'git commit -m "{commit_message}"')
    if not success:
        print(f"❌ 提交失败: {error}")
        return False
    
    # 推送到远程仓库
    print("🌐 推送到 GitHub...")
    success, _, error = run_command("git push origin main")
    if not success:
        print(f"❌ 推送失败: {error}")
        print("💡 可能的原因:")
        print("   - 网络连接问题")
        print("   - GitHub Token 过期")
        print("   - 权限不足")
        print("   - 分支名称错误")
        return False
    
    # 获取仓库 URL
    success, repo_url, _ = run_command("git remote get-url origin")
    if success:
        print(f"🔗 仓库链接: {repo_url.strip()}")
    
    print("✅ 成功推送到 GitHub!")
    print("🎉 自动推送完成!")
    return True

def main():
    """主函数"""
    try:
        # 切换到脚本所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        success = auto_push_to_github()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户取消操作")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
