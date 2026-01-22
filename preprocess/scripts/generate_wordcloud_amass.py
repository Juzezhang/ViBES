import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm
from wordcloud import WordCloud, STOPWORDS


DEFAULT_STOPWORDS = set(STOPWORDS) | {
    # domain-specific fillers
    'uh', 'um', 'like', 'you_know', 'you', 'know', 'yeah', 'ok', 'okay', 'right', 'gonna', 'wanna',
    # basic pronouns/aux
    'i', 'me', 'my', 'mine', 'we', 'our', 'ours', 'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers', 'they', 'them', 'their', 'theirs',
    'it', 'its', 'this', 'that', 'these', 'those',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'do', 'does', 'did', 'doing', "don't", "doesn't", "didn't",
    'have', 'has', 'had', 'having',
    'a', 'an', 'the', 'and', 'or', 'but',
}


def read_transcript_files(root: Path, subfolder1: str, subfolder2: str) -> tuple[list[str], dict]:
    """
    Read transcript files from two subfolders.
    Returns (all_texts, stats_dict) where stats_dict contains file counts per folder.
    """
    texts = []
    stats = {'total': 0, subfolder1: 0, subfolder2: 0}
    
    # Process both subfolders
    for subfolder in [subfolder1, subfolder2]:
        folder_path = root / subfolder
        
        if not folder_path.exists():
            print(f"Warning: {subfolder} folder not found at {folder_path}")
            continue
        
        # Find all .txt files
        txt_files = list(folder_path.rglob('*.txt'))
        print(f"Found {len(txt_files)} files in {subfolder}/")
        stats[subfolder] = len(txt_files)
        
        for txt_file in tqdm(txt_files, desc=f"Reading {subfolder}", unit="file"):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
            except UnicodeDecodeError:
                with open(txt_file, 'r', encoding='latin-1') as f:
                    texts.append(f.read())
    
    stats['total'] = len(texts)
    return texts, stats


def extract_words_from_amass_format(text: str) -> list[str]:
    """
    Extract words from AMASS_talking format.
    Format: <|assistant|> text with <|audio_xxx|> tokens
    
    We need to:
    1. Remove all special tokens like <|assistant|>, <|audio_xxx|>
    2. Extract the actual text
    3. Tokenize and filter
    """
    # Remove all special tokens: <|...|>
    cleaned_text = re.sub(r'<\|[^|]+\|>', ' ', text)
    
    # Convert to lowercase and split into words
    cleaned_text = cleaned_text.lower()
    
    # Remove extra punctuation but keep apostrophes
    cleaned_text = re.sub(r"[^\w\s\'-]", ' ', cleaned_text)
    
    # Split into tokens
    tokens = []
    for word in cleaned_text.split():
        word = word.strip("'\"").strip()
        # Filter very short words
        if word and len(word) > 2:
            tokens.append(word)
    
    return tokens


def build_frequencies(texts: list[str], extra_stopwords: set[str]) -> Counter:
    stop = DEFAULT_STOPWORDS | {w.lower() for w in extra_stopwords}
    counter: Counter = Counter()
    print(f"Processing {len(texts)} text files...")
    for t in tqdm(texts, desc="Extracting words", unit="file"):
        tokens = extract_words_from_amass_format(t)
        for token in tokens:
            if token in stop:
                continue
            # filter very short tokens already done in extract function
            counter[token] += 1
    return counter


def generate_word_cloud(counter: Counter, width: int, height: int, background_color: str) -> WordCloud:
    wc = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        prefer_horizontal=0.9,
        max_words=200,  # Reduced from 500 for less density
        min_font_size=12,  # Increase minimum font size
        relative_scaling=0.5,  # Better size distribution
        collocations=False,
    )
    wc.generate_from_frequencies(counter)
    return wc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate word cloud from AMASS_talking transcript files')
    parser.add_argument('--root', required=True, help='Root directory containing transcript folders')
    parser.add_argument('--answer_folder', default='transcripts_answer', help='Answer transcripts folder name')
    parser.add_argument('--question_folder', default='transcripts_question', help='Question transcripts folder name')
    parser.add_argument('--output', required=False, default='output/wordcloud_amass.png', help='Output PNG path')
    parser.add_argument('--width', type=int, default=2000, help='Image width')
    parser.add_argument('--height', type=int, default=1200, help='Image height')
    parser.add_argument('--bg', default='white', help='Background color')
    parser.add_argument('--stopwords', default='', help='Comma-separated extra stopwords')
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        print(f"Error: root does not exist: {root}")
        sys.exit(1)

    texts, stats = read_transcript_files(root, args.answer_folder, args.question_folder)
    if not texts:
        print('No transcript files found.')
        sys.exit(2)

    print(f"\n=== Dataset Statistics ===")
    print(f"  {args.answer_folder}: {stats[args.answer_folder]} files")
    print(f"  {args.question_folder}: {stats[args.question_folder]} files")
    print(f"  Total: {stats['total']} files\n")

    extra_stop = set([w.strip() for w in args.stopwords.split(',') if w.strip()]) if args.stopwords else set()
    freq = build_frequencies(texts, extra_stop)
    if not freq:
        print('No tokens after filtering. Consider relaxing stopwords.')
        sys.exit(3)

    print(f"Generating word cloud with {len(freq)} unique tokens...")
    wc = generate_word_cloud(freq, args.width, args.height, args.bg)
    
    print(f"Saving to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wc.to_file(str(output_path))
    
    print(f"✓ Word cloud saved successfully!")
    print(f"  Output: {args.output}")
    print(f"  Unique tokens: {len(freq)}")
    print(f"  Top 10 words: {', '.join([w for w, _ in freq.most_common(10)])}")


if __name__ == '__main__':
    main()

