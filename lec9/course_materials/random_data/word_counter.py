from collections import Counter

SAMPLE_TEXT = """
Python is a programming language that lets you work quickly and integrate
systems more effectively. Python is powerful and fast, plays well with
others, runs everywhere, is friendly and easy to learn, and is open.
"""

def count_words(text):
    words = text.lower().split()
    words = [w.strip(".,!?;:()\"'") for w in words]
    return Counter(words)

def top_words(text, n=5):
    counts = count_words(text)
    return counts.most_common(n)

def word_stats(text):
    words = text.split()
    lengths = [len(w) for w in words]
    return {
        "total_words": len(words),
        "unique_words": len(set(w.lower() for w in words)),
        "avg_length": sum(lengths) / len(lengths) if lengths else 0,
    }

if __name__ == "__main__":
    print("Top 5 words:", top_words(SAMPLE_TEXT))
    stats = word_stats(SAMPLE_TEXT)
    for k, v in stats.items():
        print(f"  {k}: {v}")
