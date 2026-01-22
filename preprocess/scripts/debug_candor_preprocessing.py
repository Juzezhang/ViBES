#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
调试版本：预处理CANDOR对话数据，生成连续的ABABABAB序列用于训练
使用方法：
    python scripts/debug_candor_preprocessing.py --data_root /path/to/CANDOR \
        --audio_dir audios_token_glm --face_dir TOKENS_DS4 \
        --output_path ./debug_output
"""

import os
import json
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path

def load_conversation_structure(structure_path):
    """加载对话结构文件"""
    try:
        with open(structure_path, 'r') as f:
            structure = json.load(f)
        print(f"已加载结构文件，包含 {len(structure)} 个对话")
        return structure
    except Exception as e:
        print(f"加载对话结构失败: {e}")
        return {}

def debug_preprocessing(data_root, audio_dir, face_dir, structure_file, 
                       output_path, audio_tokens_per_chunk=26, face_tokens_per_chunk=13,
                       max_conversations=3):
    """
    调试版本：预处理CANDOR对话数据
    """
    # 1. 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 2. 加载对话结构
    structure_path = os.path.join(data_root, structure_file)
    conversation_structure = load_conversation_structure(structure_path)
    
    # 3. 只处理有限数量的对话进行调试
    conv_ids = list(conversation_structure.keys())[:max_conversations]
    print(f"调试模式：仅处理 {len(conv_ids)} 个对话")
    
    # 4. 处理每个对话
    all_records = []
    for conv_id in conv_ids:
        print(f"\n====== 处理对话 {conv_id} ======")
        speaker_files = conversation_structure[conv_id]
        
        # 跳过少于2个说话者的对话
        if len(speaker_files) < 2:
            print(f"跳过对话 {conv_id}：说话者数量不足")
            continue
        
        # 获取说话者ID
        speaker_ids = [speaker_file.split('.')[0] for speaker_file in speaker_files]
        print(f"说话者：{speaker_ids}")
        
        # 只处理前两个说话者，A和B
        speaker_a_id = speaker_ids[0]
        speaker_b_id = speaker_ids[1]
        
        # 加载两个说话者的token
        speaker_data = []
        for speaker_id in [speaker_a_id, speaker_b_id]:
            # 音频token路径
            audio_path = os.path.join(data_root, audio_dir, conv_id, f"{speaker_id}.npy")
            if not os.path.exists(audio_path):
                print(f"未找到音频token：{audio_path}")
                continue
            
            # 面部表情token路径
            face_path = os.path.join(data_root, face_dir, conv_id, f"{speaker_id}.npy")
            if not os.path.exists(face_path):
                print(f"未找到面部token：{face_path}")
                continue
            
            # 加载token
            try:
                audio_tokens = np.load(audio_path)
                face_tokens = np.load(face_path)
                
                print(f"已加载 {speaker_id} 的token：")
                print(f"  音频token形状: {audio_tokens.shape}")
                print(f"  面部token形状: {face_tokens.shape}")
                
                # 修复面部token形状（如果需要）
                if len(face_tokens.shape) == 2 and face_tokens.shape[0] == 1:
                    face_tokens = face_tokens.T
                    print(f"  已重塑面部token到形状：{face_tokens.shape}")
                
                speaker_data.append({
                    "speaker_id": speaker_id,
                    "audio_tokens": audio_tokens,
                    "face_tokens": face_tokens
                })
            except Exception as e:
                print(f"加载 {speaker_id} 的token时出错：{e}")
                continue
        
        # 跳过数据不完整的对话
        if len(speaker_data) < 2:
            print(f"跳过对话 {conv_id}：数据不完整")
            continue
        
        print("\n===== 创建连续序列 =====")
        
        # 按块处理音频和面部token
        chunks = []
        for speaker_idx, speaker in enumerate(speaker_data):
            # 简化起见，将整个序列视为一个说话轮次
            speaker_chunks = []
            
            # 计算块的数量
            audio_tokens = speaker["audio_tokens"]
            face_tokens = speaker["face_tokens"]
            
            num_chunks = len(audio_tokens) // audio_tokens_per_chunk
            if num_chunks == 0:
                num_chunks = 1  # 至少有一个块
                
            print(f"说话者 {speaker['speaker_id']} 的块数量：{num_chunks}")
            
            # 处理每个块
            for chunk_idx in range(num_chunks):
                # 计算音频token的起始和结束索引
                audio_start = chunk_idx * audio_tokens_per_chunk
                audio_end = min(audio_start + audio_tokens_per_chunk, len(audio_tokens))
                
                # 计算面部token的起始和结束索引（一般是音频的一半频率）
                face_start = audio_start // 2
                face_end = min(face_start + face_tokens_per_chunk, len(face_tokens) if len(face_tokens.shape) == 1 else face_tokens.shape[0])
                
                # 提取块
                if audio_end - audio_start < audio_tokens_per_chunk or face_end - face_start < face_tokens_per_chunk:
                    # 跳过不完整的块
                    continue
                    
                # 提取音频和面部token
                audio_chunk = audio_tokens[audio_start:audio_end]
                
                # 提取面部token（处理不同形状的情况）
                if len(face_tokens.shape) > 1:
                    if face_tokens.shape[0] >= face_end:
                        face_chunk = face_tokens[face_start:face_end]
                    else:
                        # 不常见的情况，可能需要特殊处理
                        continue
                else:
                    face_chunk = face_tokens[face_start:face_end]
                    # 添加一个维度使其变为(N, 1)
                    face_chunk = face_chunk.reshape(-1, 1)
                
                # 创建块数据
                chunk = {
                    "chunk_id": chunk_idx,
                    "speaker_id": speaker["speaker_id"],
                    "speaker_idx": speaker_idx,
                    "audio_tokens": audio_chunk,
                    "face_tokens": face_chunk
                }
                
                speaker_chunks.append(chunk)
            
            print(f"为说话者 {speaker['speaker_id']} 创建了 {len(speaker_chunks)} 个有效块")
            chunks.extend(speaker_chunks)
        
        # 按说话者分组块
        speaker_chunks_dict = {}
        for chunk in chunks:
            speaker_id = chunk["speaker_id"]
            if speaker_id not in speaker_chunks_dict:
                speaker_chunks_dict[speaker_id] = []
            speaker_chunks_dict[speaker_id].append(chunk)
        
        # 准备A和B说话者的数据
        speaker_a_chunks = speaker_chunks_dict.get(speaker_a_id, [])
        speaker_b_chunks = speaker_chunks_dict.get(speaker_b_id, [])
        
        # 对每个说话者的块排序
        speaker_a_chunks.sort(key=lambda x: x["chunk_id"])
        speaker_b_chunks.sort(key=lambda x: x["chunk_id"])
        
        print(f"说话者A ({speaker_a_id}) 的块数量：{len(speaker_a_chunks)}")
        print(f"说话者B ({speaker_b_id}) 的块数量：{len(speaker_b_chunks)}")
        
        # 合并A和B的token
        all_speaker_a_audio = []
        all_speaker_a_face = []
        all_speaker_b_audio = []
        all_speaker_b_face = []
        
        # 将token转换为列表
        def token_to_list(array):
            if len(array.shape) > 1:
                return [int(val) for val in array.flatten()]
            else:
                return [int(val) for val in array]
        
        # 合并所有块的token
        for chunk in speaker_a_chunks:
            all_speaker_a_audio.extend(token_to_list(chunk["audio_tokens"]))
            all_speaker_a_face.extend(token_to_list(chunk["face_tokens"]))
        
        for chunk in speaker_b_chunks:
            all_speaker_b_audio.extend(token_to_list(chunk["audio_tokens"]))
            all_speaker_b_face.extend(token_to_list(chunk["face_tokens"]))
        
        print(f"合并后的token数量：")
        print(f"  说话者A的音频token：{len(all_speaker_a_audio)}")
        print(f"  说话者A的面部token：{len(all_speaker_a_face)}")
        print(f"  说话者B的音频token：{len(all_speaker_b_audio)}")
        print(f"  说话者B的面部token：{len(all_speaker_b_face)}")
        
        # 生成三种不同格式的ABAB序列
        
        # 格式1：A所有内容 + B所有内容
        format1_text = ""
        format1_text += "<|speaker_A|>"
        format1_text += "".join([f"<|audio_{val}|>" for val in all_speaker_a_audio[:100]])  # 只显示前100个token
        if len(all_speaker_a_audio) > 100:
            format1_text += "..."
        format1_text += "".join([f"<|face_{val}|>" for val in all_speaker_a_face[:100]])  # 只显示前100个token
        if len(all_speaker_a_face) > 100:
            format1_text += "..."
        
        format1_text += "<|speaker_B|>"
        format1_text += "".join([f"<|audio_{val}|>" for val in all_speaker_b_audio[:100]])  # 只显示前100个token
        if len(all_speaker_b_audio) > 100:
            format1_text += "..."
        format1_text += "".join([f"<|face_{val}|>" for val in all_speaker_b_face[:100]])  # 只显示前100个token
        if len(all_speaker_b_face) > 100:
            format1_text += "..."
        
        print("\n===== 格式1：A所有内容 + B所有内容 =====")
        print(f"前100个token示例：")
        print(format1_text[:300] + "..." if len(format1_text) > 300 else format1_text)
        
        # 格式2：A->B->A->B->...交替排列
        format2_text = ""
        min_chunks = min(len(speaker_a_chunks), len(speaker_b_chunks))
        
        for i in range(min_chunks):
            # 添加A的内容
            chunk_a = speaker_a_chunks[i]
            format2_text += "<|speaker_A|>"
            format2_text += "".join([f"<|audio_{val}|>" for val in token_to_list(chunk_a["audio_tokens"])])
            format2_text += "".join([f"<|face_{val}|>" for val in token_to_list(chunk_a["face_tokens"])])
            
            # 添加B的内容
            chunk_b = speaker_b_chunks[i]
            format2_text += "<|speaker_B|>"
            format2_text += "".join([f"<|audio_{val}|>" for val in token_to_list(chunk_b["audio_tokens"])])
            format2_text += "".join([f"<|face_{val}|>" for val in token_to_list(chunk_b["face_tokens"])])
        
        print("\n===== 格式2：A1->B1->A2->B2->...交替排列 =====")
        print(f"前300个字符示例：")
        print(format2_text[:300] + "..." if len(format2_text) > 300 else format2_text)
        
        # 格式3：音频和面部token交替出现 (A音频->A面部->B音频->B面部)
        format3_text = ""
        format3_text += "<|speaker_A|><|audio|>"
        format3_text += "".join([f"<|{val}|>" for val in all_speaker_a_audio[:100]])
        if len(all_speaker_a_audio) > 100:
            format3_text += "..."
        format3_text += "<|face|>"
        format3_text += "".join([f"<|{val}|>" for val in all_speaker_a_face[:100]])
        if len(all_speaker_a_face) > 100:
            format3_text += "..."
        
        format3_text += "<|speaker_B|><|audio|>"
        format3_text += "".join([f"<|{val}|>" for val in all_speaker_b_audio[:100]])
        if len(all_speaker_b_audio) > 100:
            format3_text += "..."
        format3_text += "<|face|>"
        format3_text += "".join([f"<|{val}|>" for val in all_speaker_b_face[:100]])
        if len(all_speaker_b_face) > 100:
            format3_text += "..."
        
        print("\n===== 格式3：音频和面部token分开标记 =====")
        print(f"前300个字符示例：")
        print(format3_text[:300] + "..." if len(format3_text) > 300 else format3_text)
        
        # 创建记录
        record = {
            "id": len(all_records),
            "conv_id": conv_id,
            "speaker_a_id": speaker_a_id,
            "speaker_b_id": speaker_b_id,
            "num_chunks_a": len(speaker_a_chunks),
            "num_chunks_b": len(speaker_b_chunks),
            "speaker_a_audio": all_speaker_a_audio,
            "speaker_a_face": all_speaker_a_face,
            "speaker_b_audio": all_speaker_b_audio,
            "speaker_b_face": all_speaker_b_face,
            "format1_text": format1_text,
            "format2_text": format2_text,
            "format3_text": format3_text
        }
        
        all_records.append(record)
    
    # 保存处理后的数据
    print(f"\n===== 保存 {len(all_records)} 个对话 =====")
    
    # 以JSONL格式保存数据（行式JSON，便于加载）
    jsonl_path = os.path.join(output_path, "candor_debug.jsonl")
    with open(jsonl_path, "w") as f:
        for record in all_records:
            # 创建一个不含序列完整token的轻量级版本用于保存
            save_record = {
                "id": record["id"],
                "conv_id": record["conv_id"],
                "speaker_a_id": record["speaker_a_id"],
                "speaker_b_id": record["speaker_b_id"],
                "num_chunks_a": record["num_chunks_a"],
                "num_chunks_b": record["num_chunks_b"],
                # 保存token的总数而不是完整的列表
                "speaker_a_audio_len": len(record["speaker_a_audio"]),
                "speaker_a_face_len": len(record["speaker_a_face"]),
                "speaker_b_audio_len": len(record["speaker_b_audio"]),
                "speaker_b_face_len": len(record["speaker_b_face"]),
                # 保存文本格式的样本（前300个字符）
                "format1_text_sample": record["format1_text"][:300],
                "format2_text_sample": record["format2_text"][:300],
                "format3_text_sample": record["format3_text"][:300],
                # 记录总长度
                "format1_text_len": len(record["format1_text"]),
                "format2_text_len": len(record["format2_text"]),
                "format3_text_len": len(record["format3_text"])
            }
            f.write(json.dumps(save_record) + "\n")
    
    print(f"已保存调试数据到 {jsonl_path}")
    
    # 保存元数据
    metadata = {
        "processed_conversations": len(all_records),
        "audio_tokens_per_chunk": audio_tokens_per_chunk,
        "face_tokens_per_chunk": face_tokens_per_chunk,
        "available_formats": ["format1_text", "format2_text", "format3_text"],
    }
    
    with open(os.path.join(output_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"已保存元数据到 {os.path.join(output_path, 'metadata.json')}")
    
    # 保存完整的第一个记录（用于详细检查）
    if all_records:
        full_record_path = os.path.join(output_path, "full_sample.json")
        with open(full_record_path, "w") as f:
            json.dump(all_records[0], f, indent=2)
        print(f"已保存完整的第一个样本到 {full_record_path}")
    
    return all_records

def main():
    parser = argparse.ArgumentParser(description="调试CANDOR对话数据预处理")
    parser.add_argument("--data_root", type=str, required=True, help="数据集根目录")
    parser.add_argument("--audio_dir", type=str, required=True, help="音频token目录")
    parser.add_argument("--face_dir", type=str, required=True, help="面部token目录")
    parser.add_argument("--structure_file", type=str, default="candor_structure.json", help="对话结构文件")
    parser.add_argument("--audio_tokens_per_chunk", type=int, default=26, help="每个块的音频token数量")
    parser.add_argument("--face_tokens_per_chunk", type=int, default=13, help="每个块的面部token数量")
    parser.add_argument("--output_path", type=str, required=True, help="输出目录")
    parser.add_argument("--max_conversations", type=int, default=3, help="要处理的最大对话数量")
    
    args = parser.parse_args()
    
    print("===== 开始调试CANDOR数据预处理 =====")
    print(f"数据根目录: {args.data_root}")
    print(f"音频目录: {args.audio_dir}")
    print(f"面部目录: {args.face_dir}")
    print(f"每个块的音频token数量: {args.audio_tokens_per_chunk}")
    print(f"每个块的面部token数量: {args.face_tokens_per_chunk}")
    print(f"最大对话数量: {args.max_conversations}")
    
    # 调试预处理
    debug_preprocessing(
        args.data_root, 
        args.audio_dir, 
        args.face_dir, 
        args.structure_file,
        args.output_path,
        args.audio_tokens_per_chunk,
        args.face_tokens_per_chunk,
        args.max_conversations
    )
    
    print("\n===== 调试完成 =====")
    print(f"输出保存到: {args.output_path}")
    print(f"请检查生成的JSON文件以查看详细内容")

if __name__ == "__main__":
    main() 