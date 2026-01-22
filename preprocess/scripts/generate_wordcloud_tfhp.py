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


def read_transcript_files(root: Path) -> list[str]:
    texts: list[str] = []
    # First collect all .txt file paths recursively
    paths = list(root.rglob('*.txt'))
    print(f"Found {len(paths)} transcript files")
    
    for path in tqdm(paths, desc="Reading files", unit="file"):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
        except UnicodeDecodeError:
            with open(path, 'r', encoding='latin-1') as f:
                texts.append(f.read())
    return texts


def extract_words_from_whisper_output(text: str) -> list[str]:
    """
    Extract words from Whisper output format.
    Format:
        Segment X:
        Timestamp: ...
        Text: ...
        Words:
         word1: timestamp - timestamp
         word2: timestamp - timestamp
    """
    tokens = []
    lines = text.split('\n')
    in_words_section = False
    
    for line in lines:
        # Check if we're entering a Words section
        if line.strip() == 'Words:':
            in_words_section = True
            continue
        
        # Check if we're leaving Words section (empty line or new Segment)
        if in_words_section and (line.strip() == '' or line.startswith('Segment')):
            in_words_section = False
            continue
        
        # Extract word if we're in Words section
        if in_words_section:
            # Format: " word: timestamp - timestamp"
            match = re.match(r'\s*([^:]+):\s*[\d.]+s\s*-\s*[\d.]+s', line)
            if match:
                word = match.group(1).strip()
                # Remove punctuation from word but keep the base form
                word = re.sub(r'[^\w\s\'-]', '', word).lower().strip()
                if word and len(word) > 2:  # Filter very short words
                    tokens.append(word)
    
    return tokens


def build_frequencies(texts: list[str], extra_stopwords: set[str]) -> Counter:
    stop = DEFAULT_STOPWORDS | {w.lower() for w in extra_stopwords}
    counter: Counter = Counter()
    print(f"Processing {len(texts)} text files...")
    for t in tqdm(texts, desc="Extracting words", unit="file"):
        tokens = extract_words_from_whisper_output(t)
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
    parser = argparse.ArgumentParser(description='Generate word cloud from TFHP transcript files')
    parser.add_argument('--root', required=True, help='Root directory containing transcript folders')
    parser.add_argument('--output', required=False, default='output/wordcloud_tfhp.png', help='Output PNG path')
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

    texts = read_transcript_files(root)
    if not texts:
        print('No transcript files found.')
        sys.exit(2)

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

