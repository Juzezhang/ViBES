#!/usr/bin/env python3
"""
智能重复视频检测工具
基于文件大小精确匹配和内容哈希的混合方法
"""

import os
import hashlib
from pathlib import Path
from collections import defaultdict
import argparse
from tqdm import tqdm
import time

def calculate_file_hash(file_path, chunk_size=8192):
    """计算文件的MD5哈希值"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"计算哈希时出错 {file_path}: {e}")
        return None

def calculate_partial_hash(file_path, max_bytes=1024*1024):
    """计算文件的部分哈希（前1MB）用于快速预筛选"""
    try:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            chunk = f.read(max_bytes)
            hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except:
        return None

def find_duplicate_videos_smart(directory_path, min_size_mb=1.0, use_partial_hash=True):
    """
    智能重复视频检测
    
    Args:
        directory_path: 视频目录路径
        min_size_mb: 最小文件大小（MB）
        use_partial_hash: 是否使用部分哈希进行预筛选
    """
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    
    print(f"扫描目录: {directory_path}")
    print(f"最小文件大小: {min_size_mb}MB")
    print(f"使用部分哈希预筛选: {use_partial_hash}")
    print("-" * 50)
    
    if not os.path.exists(directory_path):
        print(f"错误: 目录不存在: {directory_path}")
        return {}
    
    # 收集所有视频文件
    video_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if Path(file).suffix.lower() in video_extensions:
                video_files.append(Path(root) / file)
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    if len(video_files) == 0:
        print("没有找到视频文件")
        return {}
    
    # 按文件大小分组
    size_to_files = defaultdict(list)
    for video_path in video_files:
        try:
            size = os.path.getsize(video_path)
            size_mb = size / (1024 * 1024)
            if size_mb >= min_size_mb:
                size_to_files[size].append(video_path)
        except:
            continue
    
    # 统计信息
    total_filtered_files = sum(len(files) for files in size_to_files.values())
    print(f"过滤后剩余 {total_filtered_files} 个文件")
    
    # 只处理有多个文件的组
    size_groups = [(size, files) for size, files in size_to_files.items() if len(files) > 1]
    print(f"需要比较的组数: {len(size_groups)}")
    print(f"可能重复的文件数: {sum(len(files) for _, files in size_groups)}")
    
    if not size_groups:
        print("没有找到需要比较的文件组")
        return {}
    
    duplicates = {}
    
    # 方法1: 精确大小匹配（高置信度）
    print("\n=== 方法1: 精确大小匹配 ===")
    exact_size_duplicates = 0
    
    for size, files in tqdm(size_groups, desc="检查精确大小匹配"):
        if len(files) > 1:
            # 如果文件大小完全一样，高概率是重复文件
            duplicates[f"exact_size_{size}"] = files
            exact_size_duplicates += len(files) - 1
            print(f"发现 {len(files)} 个相同大小的文件 ({size} bytes):")
            for file_path in files:
                size_mb = size / (1024 * 1024)
                print(f"  - {file_path} ({size_mb:.2f} MB)")
    
    print(f"精确大小匹配找到 {exact_size_duplicates} 个重复文件")
    
    # 方法2: 部分哈希预筛选 + 完整哈希验证
    if use_partial_hash:
        print("\n=== 方法2: 部分哈希 + 完整哈希验证 ===")
        partial_hash_duplicates = 0
        
        for size, files in tqdm(size_groups, desc="计算部分哈希"):
            if len(files) > 1:
                # 计算部分哈希
                partial_hash_groups = defaultdict(list)
                for file_path in files:
                    partial_hash = calculate_partial_hash(file_path)
                    if partial_hash:
                        partial_hash_groups[partial_hash].append(file_path)
                
                # 检查部分哈希相同的文件
                for partial_hash, hash_files in partial_hash_groups.items():
                    if len(hash_files) > 1:
                        print(f"部分哈希匹配 ({len(hash_files)} 个文件):")
                        for file_path in hash_files:
                            size_mb = os.path.getsize(file_path) / (1024 * 1024)
                            print(f"  - {file_path} ({size_mb:.2f} MB)")
                        
                        # 计算完整哈希进行最终确认
                        full_hash_groups = defaultdict(list)
                        for file_path in hash_files:
                            full_hash = calculate_file_hash(file_path)
                            if full_hash:
                                full_hash_groups[full_hash].append(file_path)
                        
                        # 记录完整哈希相同的文件
                        for full_hash, full_hash_files in full_hash_groups.items():
                            if len(full_hash_files) > 1:
                                duplicates[f"full_hash_{full_hash[:16]}"] = full_hash_files
                                partial_hash_duplicates += len(full_hash_files) - 1
                                print(f"  完整哈希确认重复 ({len(full_hash_files)} 个文件)")
    
    print(f"部分哈希方法找到 {partial_hash_duplicates} 个重复文件")
    
    return duplicates

def print_duplicates(duplicates):
    """打印重复视频信息"""
    if not duplicates:
        print("没有找到重复的视频文件")
        return
    
    print(f"\n找到 {len(duplicates)} 组重复视频:")
    print("=" * 80)
    
    total_duplicates = 0
    total_size_saved = 0
    
    for i, (group_id, video_list) in enumerate(duplicates.items(), 1):
        print(f"\n重复组 {i} (ID: {group_id}):")
        print("-" * 40)
        
        # 按文件大小排序，通常较大的文件是原始文件
        video_list_with_size = []
        for video_path in video_list:
            try:
                size = os.path.getsize(video_path)
                video_list_with_size.append((video_path, size))
            except:
                video_list_with_size.append((video_path, 0))
        
        video_list_with_size.sort(key=lambda x: x[1], reverse=True)
        
        for j, (video_path, size) in enumerate(video_list_with_size):
            size_mb = size / (1024 * 1024)
            status = " (建议保留)" if j == 0 else " (建议删除)"
            print(f"  {j+1}. {video_path} ({size_mb:.2f} MB){status}")
            if j > 0:  # 计算可节省的空间
                total_size_saved += size
        
        total_duplicates += len(video_list) - 1  # 减去保留的文件
    
    print(f"\n总结:")
    print(f"可删除的重复文件数: {total_duplicates}")
    print(f"可节省的空间: {total_size_saved / (1024 * 1024):.2f} MB")

def save_duplicates_report(duplicates, output_file):
    """保存重复视频报告到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("智能重复视频检测报告\n")
        f.write("=" * 50 + "\n\n")
        
        if not duplicates:
            f.write("没有找到重复的视频文件\n")
            return
        
        f.write(f"找到 {len(duplicates)} 组重复视频:\n\n")
        
        total_duplicates = 0
        total_size_saved = 0
        
        for i, (group_id, video_list) in enumerate(duplicates.items(), 1):
            f.write(f"重复组 {i} (ID: {group_id}):\n")
            f.write("-" * 40 + "\n")
            
            # 按文件大小排序
            video_list_with_size = []
            for video_path in video_list:
                try:
                    size = os.path.getsize(video_path)
                    video_list_with_size.append((video_path, size))
                except:
                    video_list_with_size.append((video_path, 0))
            
            video_list_with_size.sort(key=lambda x: x[1], reverse=True)
            
            for j, (video_path, size) in enumerate(video_list_with_size):
                size_mb = size / (1024 * 1024)
                status = " (建议保留)" if j == 0 else " (建议删除)"
                f.write(f"  {j+1}. {video_path} ({size_mb:.2f} MB){status}\n")
                if j > 0:
                    total_size_saved += size
            
            f.write("\n")
            total_duplicates += len(video_list) - 1
        
        f.write(f"\n总结:\n")
        f.write(f"可删除的重复文件数: {total_duplicates}\n")
        f.write(f"可节省的空间: {total_size_saved / (1024 * 1024):.2f} MB\n")

def main():
    parser = argparse.ArgumentParser(description='智能检测视频目录中的重复文件')
    parser.add_argument('directory', help='要扫描的视频目录路径')
    parser.add_argument('--min-size', type=float, default=1.0,
                       help='最小文件大小，MB (默认: 1.0)')
    parser.add_argument('--output', '-o', help='输出报告文件路径')
    parser.add_argument('--no-partial-hash', action='store_true',
                       help='禁用部分哈希预筛选')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # 查找重复视频
    duplicates = find_duplicate_videos_smart(
        args.directory, 
        args.min_size,
        not args.no_partial_hash
    )
    
    end_time = time.time()
    print(f"\n处理完成，耗时: {end_time - start_time:.2f} 秒")
    
    # 打印结果
    print_duplicates(duplicates)
    
    # 保存报告
    if args.output:
        save_duplicates_report(duplicates, args.output)
        print(f"\n报告已保存到: {args.output}")

if __name__ == "__main__":
    main()
