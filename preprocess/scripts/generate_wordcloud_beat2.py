import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm
from wordcloud import WordCloud, STOPWORDS

try:
    import textgrid as tg
except ImportError:
    print("Error: textgrid library not found. Install with: pip install textgrid")
    sys.exit(1)


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


def read_textgrid_files(root: Path) -> list[list[str]]:
    """
    Read all TextGrid files and extract words from each.
    Returns a list of word lists (one list per file).
    """
    textgrid_files = list(root.rglob('*.TextGrid'))
    print(f"Found {len(textgrid_files)} TextGrid files")
    
    all_words = []
    failed_count = 0
    
    for tg_file in tqdm(textgrid_files, desc="Reading TextGrid files", unit="file"):
        try:
            tgrid = tg.TextGrid.fromFile(str(tg_file))
            
            # Extract words from the first tier (typically 'words' tier)
            file_words = []
            
            if len(tgrid) > 0:
                word_tier = tgrid[0]  # First tier contains word-level annotations
                for interval in word_tier:
                    word_text = interval.mark.strip()
                    
                    # Skip empty intervals
                    if word_text and word_text != "":
                        # Clean the word text
                        cleaned_word = clean_word(word_text)
                        if cleaned_word:
                            file_words.append(cleaned_word)
            
            if file_words:
                all_words.append(file_words)
                
        except Exception as e:
            failed_count += 1
            # Silently continue, just count failures
            continue
    
    if failed_count > 0:
        print(f"Warning: Failed to read {failed_count} TextGrid files")
    
    return all_words


def clean_word(word: str) -> str:
    """
    Clean a word from TextGrid annotation.
    Remove punctuation, convert to lowercase, filter short words.
    """
    # Remove special characters and punctuation
    word = re.sub(r'[^\w\s\'-]', '', word).lower().strip()
    
    # Remove leading/trailing quotes and apostrophes
    word = word.strip("'\"")
    
    # Filter very short words
    if len(word) <= 2:
        return ""
    
    return word


def build_frequencies(word_lists: list[list[str]], extra_stopwords: set[str]) -> Counter:
    stop = DEFAULT_STOPWORDS | {w.lower() for w in extra_stopwords}
    counter: Counter = Counter()
    
    print(f"Processing {len(word_lists)} files...")
    for words in tqdm(word_lists, desc="Counting words", unit="file"):
        for word in words:
            if word in stop:
                continue
            counter[word] += 1
    
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
    parser = argparse.ArgumentParser(description='Generate word cloud from BEAT2 TextGrid files')
    parser.add_argument('--root', required=True, help='Root directory containing TextGrid files')
    parser.add_argument('--output', required=False, default='output/wordcloud_beat2.png', help='Output PNG path')
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

    word_lists = read_textgrid_files(root)
    if not word_lists:
        print('No TextGrid files found or all failed to parse.')
        sys.exit(2)

    extra_stop = set([w.strip() for w in args.stopwords.split(',') if w.strip()]) if args.stopwords else set()
    freq = build_frequencies(word_lists, extra_stop)
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

