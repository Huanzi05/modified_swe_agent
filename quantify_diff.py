import json
import os
import difflib
import shlex
from collections import Counter

def load_traj(file_path):
    """加载 .traj 文件"""
    if not os.path.exists(file_path):
        print(f" 找不到文件: {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        traj_data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    traj_data.append(json.loads(line))
        return traj_data

def extract_commands_and_files(trajectory_data):
    """提取命令序列和关注的文件集合"""
    commands = []
    files_touched = set()
    
    steps = trajectory_data.get('trajectory', trajectory_data) if isinstance(trajectory_data, dict) else trajectory_data
    
    for step in steps:
        action = step.get('action', step.get('response', ''))
        if not action and 'info' in step and isinstance(step['info'], dict):
            action = step['info'].get('action', '')
            
        if not isinstance(action, str) or not action.strip():
            continue

        # 尝试解析 Function Calling (JSON) 格式
        try:
            call = json.loads(action)
            if isinstance(call, dict) and "name" in call:
                cmd_name = call["name"]
                args = call.get("arguments", {})
                
                if cmd_name == "str_replace_editor":
                    commands.append(f"str_replace_editor:{args.get('command', 'unknown')}")
                    if "path" in args:
                        files_touched.add(args["path"])
                    continue
                elif cmd_name == "bash":
                    action = args.get("command", "") # 提取 bash 命令字符串继续向下解析
                elif cmd_name == "submit":
                    commands.append("submit")
                    continue
        except json.JSONDecodeError:
            pass # 不是 JSON，按普通字符串解析

        # 解析纯文本命令字符串
        lines = action.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('```') and not line.startswith('#'):
                try:
                    parts = shlex.split(line) 
                except ValueError:
                    parts = line.split()
                    
                if not parts:
                    continue
                    
                cmd = parts[0]
                commands.append(cmd)
                
                # 1. 处理 SWE-agent 专属编辑器命令: str_replace_editor <command> <path>
                if cmd == 'str_replace_editor' and len(parts) > 2:
                    file_path = parts[2].strip('\'"')
                    files_touched.add(file_path)
                    
                break 
                
    return commands, files_touched

def calculate_metrics(file1, file2, name1="Identity Run", name2="Paraphrase Run"):
    traj1 = load_traj(file1)
    traj2 = load_traj(file2)

    if not traj1 or not traj2:
        return

    steps1 = traj1.get('trajectory', traj1) if isinstance(traj1, dict) else traj1
    steps2 = traj2.get('trajectory', traj2) if isinstance(traj2, dict) else traj2
    
    len1, len2 = len(steps1), len(steps2)
    
    divergence_step = -1
    for i in range(min(len1, len2)):
        a1 = steps1[i].get('action', steps1[i].get('response', ''))
        a2 = steps2[i].get('action', steps2[i].get('response', ''))
        if a1 != a2:
            divergence_step = i
            break
    if divergence_step == -1:
        divergence_step = min(len1, len2)

    cmds1, set_files1 = extract_commands_and_files(traj1)
    cmds2, set_files2 = extract_commands_and_files(traj2)

    matcher = difflib.SequenceMatcher(None, cmds1, cmds2)
    sequence_similarity = matcher.ratio()

    intersection = len(set_files1.intersection(set_files2))
    union = len(set_files1.union(set_files2))
    file_overlap_ratio = intersection / union if union > 0 else 0.0

    freq1 = Counter(cmds1)
    freq2 = Counter(cmds2)

    print("\n" + "="*50)
    print("📊 SWE-agent 轨迹差异量化分析报告")
    print("="*50)
    
    print("\n[1] 长度与分岔点")
    print(f"  - {name1} 总步数: {len1}")
    print(f"  - {name2} 总步数: {len2}")
    print(f"  - 首次分岔点: 第 {divergence_step} 步 (Step {divergence_step})")
    
    print("\n[2] 命令序列相似度")
    print(f"  - difflib.SequenceMatcher Ratio: {sequence_similarity:.2%} (100% 为完全一致)")
    
    print("\n[3] 文件关注重合度 (Jaccard Index)")
    print(f"  - {name1} 关注文件数: {len(set_files1)}")
    if len(set_files1) > 0:
        print(f"    (文件列表: {', '.join(list(set_files1))}{'...' if len(set_files1)>3 else ''})")
        
    print(f"  - {name2} 关注文件数: {len(set_files2)}")
    if len(set_files2) > 0:
        print(f"    (文件列表: {', '.join(list(set_files2))}{'...' if len(set_files2)>3 else ''})")
        
    print(f"  - 交集文件数: {intersection}")
    print(f"  - 重合度 (交并比): {file_overlap_ratio:.2%} (100% 为查看了完全相同的文件集合)")
    
    print("\n[4] 行为偏好 (命令频次 Top 5)")
    print(f"  - {name1}:")
    for cmd, count in freq1.most_common(5):
        print(f"      * {cmd:<18} : {count} 次")
        
    print(f"  - {name2}:")
    for cmd, count in freq2.most_common(5):
        print(f"      * {cmd:<18} : {count} 次")
        
    print("\n" + "="*50)

if __name__ == "__main__":
    path_identity = "trajectories/identity_verified_run/astropy__astropy-14365/astropy__astropy-14365.traj"
    path_paraphrase = "trajectories/paraphrase_verified_run/astropy__astropy-14365/astropy__astropy-14365.traj"
    
    calculate_metrics(path_identity, path_paraphrase)