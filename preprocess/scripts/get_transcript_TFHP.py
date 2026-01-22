import lmdb
import pickle
import os
import io
import soundfile as sf
import numpy as np
import torch
from tqdm import tqdm
import librosa
from faster_whisper import WhisperModel

def process_audio(audio_data, sample_rate):
    # 确保音频是float32类型，并且在[-1, 1]范围内
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    
    if audio_data.max() > 1.0 or audio_data.min() < -1.0:
        audio_data = audio_data / max(abs(audio_data.max()), abs(audio_data.min()))
    
    # 如果是立体声，转换为单声道
    if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
        audio_data = audio_data.mean(axis=1)
    
    # 重采样到16kHz（Whisper模型需要的采样率）
    if sample_rate != 16000:
        audio_data = librosa.resample(
            y=audio_data, 
            orig_sr=sample_rate, 
            target_sr=16000
        )
    
    return audio_data

def get_transcription_with_timestamps(model, audio_data):
    # 使用Whisper模型进行转录
    segments, info = model.transcribe(
        audio_data,
        task="transcribe",
        language="en",
        word_timestamps=True  # 启用单词级时间戳
    )
    
    return list(segments), info

def save_transcript(output_file, segments, info):
    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入完整文本
        full_text = ' '.join(segment.text for segment in segments)
        f.write(f"Full text: {full_text}\n\n")
        
        # 写入语言和检测信息
        f.write(f"Detected language: {info.language} (probability: {info.language_probability:.3f})\n\n")
        
        # 写入详细的分段信息
        f.write("Segments with word-level timestamps:\n")
        for i, segment in enumerate(segments, 1):
            f.write(f"\nSegment {i}:\n")
            f.write(f"Timestamp: {segment.start:.3f}s - {segment.end:.3f}s\n")
            f.write(f"Text: {segment.text}\n")
            if hasattr(segment, 'confidence'):
                f.write(f"Confidence: {segment.confidence:.3f}\n")
            f.write("Words:\n")
            
            if hasattr(segment, 'words'):
                for word in segment.words:
                    f.write(f"{word.word}: {word.start:.3f}s - {word.end:.3f}s")
                    if hasattr(word, 'confidence'):
                        f.write(f" (confidence: {word.confidence:.3f})")
                    f.write("\n")
            
            f.write("-" * 50 + "\n")

def save_coef(output_file, coef_data):
    # 保存coef数据为npz格式
    np.savez(output_file, **coef_data)

def main():
    # 加载Whisper模型
    print("Loading Whisper model...")
    model = WhisperModel("medium", device="cpu", compute_type="int8")
    
    # LMDB 路径
    # lmdb_path = "/simurgh/u/juze/code/conversational_agent/datasets/TFHP/HDTF_TFHP-lmdb"
    lmdb_path = "/simurgh/u/juze/datasets/TFHP/HDTF_TFHP-lmdb"

    # 打开 LMDB
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    
    # 创建输出目录
    transcript_dir = "/simurgh/u/juze/datasets/TFHP/transcripts"
    coef_dir = "/simurgh/u/juze/datasets/TFHP/coef"
    os.makedirs(transcript_dir, exist_ok=True)
    os.makedirs(coef_dir, exist_ok=True)
    
    # 创建日志文件
    log_file = os.path.join(transcript_dir, "processing_log.txt")
    
    with env.begin() as txn:
        cursor = txn.cursor()
        total_samples = sum(1 for _ in cursor)  # 计算总样本数
        cursor = txn.cursor()  # 重置游标
        
        # 使用tqdm显示进度
        with tqdm(total=total_samples, desc="Processing audio files") as pbar:
            for key, value in cursor:
                try:
                    # 获取原始文件名（从key中解码）
                    sequence_name = key.decode('utf-8')
                    # if sequence_name != 'TH_00269/003/003':
                    #     continue
                    if sequence_name.split('/')[-1] == 'metadata':
                        continue
                    # 解析数据
                    data = pickle.loads(value)
                    audio_bytes = data['audio']
                    coef_data = data['coef']
                    
                    # 将字节数据转换为音频数组
                    audio_io = io.BytesIO(audio_bytes)
                    audio, sample_rate = sf.read(audio_io)
                    
                    # 处理音频数据
                    processed_audio = process_audio(audio, sample_rate)
                    
                    # 获取转录文本和时间戳
                    segments, info = get_transcription_with_timestamps(model, processed_audio)
                    
                    # 保存转录结果
                    transcript_file = os.path.join(transcript_dir, f"{sequence_name}.txt")
                    os.makedirs(os.path.dirname(transcript_file), exist_ok=True)
                    save_transcript(transcript_file, segments, info)
                    
                    # 保存coef数据
                    coef_file = os.path.join(coef_dir, f"{sequence_name}.npz")
                    os.makedirs(os.path.dirname(coef_file), exist_ok=True)
                    save_coef(coef_file, coef_data)
                    
                    # 保存音频数据（wav格式）
                    audio_file = os.path.join("/simurgh/u/juze/datasets/TFHP/audios", f"{sequence_name}.wav")
                    os.makedirs(os.path.dirname(audio_file), exist_ok=True)
                    sf.write(audio_file, audio, sample_rate)
                    
                    # 更新进度条
                    pbar.update(1)
                    
                except Exception as e:
                    # # 记录错误到日志文件
                    # with open(log_file, 'a', encoding='utf-8') as f:
                    #     f.write(f"Error processing sequence {sequence_name}: {str(e)}\n")
                    # if sequence_name == 'TH_00269/003/003':
                    #     pass
                    print(f"Error processing sequence {sequence_name}: {str(e)}")
                    continue
    
    print(f"\nProcessing complete.")
    print(f"Transcripts saved in {transcript_dir}")
    print(f"Coef data saved in {coef_dir}")
    print(f"Check {log_file} for any processing errors.")

if __name__ == "__main__":
    main()
