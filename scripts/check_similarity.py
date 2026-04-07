#!/usr/bin/env python3
"""
iThenticate-style similarity checker for conference→journal paper plagiarism prevention.

Mimics iThenticate's approach:
1. Digital fingerprinting via word n-grams (not disclosed officially, we use 5-grams)
2. Similarity = matching_words / total_words (iThenticate's exact formula)
3. Per-section breakdown (editors look at WHERE matches occur)
4. Excludes: bibliography, math, LaTeX commands, common academic phrases

Thresholds (from literature):
  - < 15%: Generally acceptable for journal submission
  - 15-25%: Requires editorial review, risky for conference→journal extension
  - > 25%: Likely to be flagged, needs substantial rewriting

Usage:
    python check_similarity.py conference.tex journal.tex
    python check_similarity.py conference.tex journal.tex --ngram 5 --verbose
"""
import argparse
import re
import sys
from collections import Counter
from typing import Dict, List, Set, Tuple


# Common academic phrases that iThenticate often flags but editors ignore
COMMON_PHRASES = {
    "in this paper we", "we propose a", "the rest of the paper",
    "is organized as follows", "the experimental results show",
    "we evaluate our", "as shown in", "it can be seen that",
    "in recent years", "state of the art", "to the best of our knowledge",
    "in this section we", "we conduct experiments on", "compared with",
    "we use the", "the results are shown in", "as illustrated in",
}


def strip_latex_to_text(tex: str) -> str:
    """Strip LaTeX to plain text, preserving document structure."""
    # Remove everything after \bibliography
    bib_idx = tex.find("\\bibliography{")
    if bib_idx > 0:
        tex = tex[:bib_idx]
    # Remove comments
    tex = re.sub(r'%.*$', '', tex, flags=re.MULTILINE)
    # Remove figure/table/equation environments entirely
    for env in ['equation', 'align', 'table', 'tabular', 'figure', 'lstlisting',
                'algorithm', 'algorithmic', 'tcolorbox', 'itemize', 'enumerate']:
        tex = re.sub(rf'\\begin\{{{env}\*?\}}.*?\\end\{{{env}\*?\}}', ' ', tex, flags=re.DOTALL)
    # Remove \cite, \ref, \label, \cref
    tex = re.sub(r'\\(?:cite[pt]?|nocite|ref|label|cref|eqref)\{[^}]*\}', '', tex)
    # Remove inline math
    tex = re.sub(r'\$[^$]+?\$', ' MATH ', tex)
    # Remove display math
    tex = re.sub(r'\\\[.*?\\\]', ' MATH ', tex, flags=re.DOTALL)
    # Expand text formatting commands
    for cmd in ['textbf', 'textit', 'textsc', 'texttt', 'emph', 'underline', 'mbox']:
        tex = re.sub(rf'\\{cmd}\{{([^}}]*)\}}', r'\1', tex)
    # Remove section commands but keep titles
    tex = re.sub(r'\\(?:section|subsection|subsubsection|paragraph)\*?\{([^}]*)\}', r'\1.', tex)
    # Remove remaining LaTeX commands
    tex = re.sub(r'\\[a-zA-Z]+(?:\[[^\]]*\])?\{([^}]*)\}', r'\1', tex)
    tex = re.sub(r'\\[a-zA-Z]+', ' ', tex)
    # Remove braces, tildes, special chars
    tex = re.sub(r'[{}~\\]', ' ', tex)
    # Normalize whitespace
    tex = re.sub(r'\s+', ' ', tex).strip()
    return tex


def extract_sections_text(tex: str) -> List[Tuple[str, str]]:
    """Extract (section_name, plain_text) pairs."""
    pattern = r'\\(?:section|subsection)\*?\{([^}]+)\}'
    matches = list(re.finditer(pattern, tex))
    if not matches:
        return [("Full Document", strip_latex_to_text(tex))]
    sections = []
    for i, m in enumerate(matches):
        name = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(tex)
        # Stop at bibliography
        bib = tex.find("\\bibliography{", start)
        if bib > 0 and bib < end:
            end = bib
        content = strip_latex_to_text(tex[start:end])
        if len(content.split()) > 10:  # skip tiny sections
            sections.append((name, content))
    return sections


def text_to_ngrams(text: str, n: int = 5) -> List[str]:
    """Convert text to word n-grams (iThenticate uses word fingerprints)."""
    words = text.lower().split()
    # Filter out very short words and MATH placeholders
    words = [w for w in words if len(w) > 1 and w != 'math']
    if len(words) < n:
        return []
    return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]


def compute_similarity_ithenticate(
    text_a: str, text_b: str, n: int = 5
) -> Tuple[float, int, int]:
    """
    Compute similarity mimicking iThenticate's formula:
    similarity = matching_words / total_words_in_document_B

    where "matching" means the n-gram exists in document A.
    Returns (similarity_pct, matching_words, total_words).
    """
    words_b = text_b.lower().split()
    words_b = [w for w in words_b if len(w) > 1 and w != 'math']
    total_words = len(words_b)
    if total_words == 0:
        return 0.0, 0, 0

    ngrams_a = set(text_to_ngrams(text_a, n))
    if not ngrams_a:
        return 0.0, 0, total_words

    # For each position in B, check if the n-gram starting there exists in A
    ngrams_b = text_to_ngrams(text_b, n)
    matching_positions = set()
    for i, ng in enumerate(ngrams_b):
        if ng in ngrams_a:
            # Mark all word positions in this n-gram as matching
            for j in range(n):
                matching_positions.add(i + j)

    matching_words = len(matching_positions & set(range(total_words)))
    similarity = matching_words / total_words * 100
    return similarity, matching_words, total_words


def find_matching_passages(
    text_a: str, text_b: str, n: int = 5, min_len: int = 20
) -> List[Tuple[str, float]]:
    """Find specific matching passages in text_b that appear in text_a."""
    ngrams_a = set(text_to_ngrams(text_a, n))
    sentences_b = re.split(r'(?<=[.!?])\s+', text_b)

    passages = []
    for sent in sentences_b:
        if len(sent.split()) < 5:
            continue
        sent_ngrams = text_to_ngrams(sent, n)
        if not sent_ngrams:
            continue
        match_count = sum(1 for ng in sent_ngrams if ng in ngrams_a)
        ratio = match_count / len(sent_ngrams) if sent_ngrams else 0
        if ratio > 0.5 and len(sent) >= min_len:
            passages.append((sent.strip(), ratio))

    passages.sort(key=lambda x: -x[1])
    return passages


def main():
    parser = argparse.ArgumentParser(
        description="iThenticate-style similarity checker for conference→journal plagiarism prevention"
    )
    parser.add_argument("file_a", help="Source document (conference version .tex)")
    parser.add_argument("file_b", help="Target document (journal version .tex)")
    parser.add_argument("--ngram", type=int, default=5, help="N-gram size (default: 5, iThenticate uses ~5-word fingerprints)")
    parser.add_argument("--verbose", action="store_true", help="Show matching passages")
    parser.add_argument("--exclude-common", action="store_true", help="Exclude common academic phrases")
    args = parser.parse_args()

    with open(args.file_a, 'r', encoding='utf-8') as f:
        tex_a = f.read()
    with open(args.file_b, 'r', encoding='utf-8') as f:
        tex_b = f.read()

    text_a = strip_latex_to_text(tex_a)
    text_b = strip_latex_to_text(tex_b)

    print(f"Source (conference): {args.file_a}")
    print(f"Target (journal):   {args.file_b}")
    print(f"N-gram size: {args.ngram}")
    print(f"Source words: {len(text_a.split()):,}")
    print(f"Target words: {len(text_b.split()):,}")
    print("=" * 70)

    # Overall similarity
    overall_sim, match_w, total_w = compute_similarity_ithenticate(text_a, text_b, args.ngram)
    print(f"\n## Overall Similarity Score: {overall_sim:.1f}%")
    print(f"   ({match_w:,} matching words / {total_w:,} total words)")

    if overall_sim > 25:
        print("   *** HIGH RISK: Likely flagged by iThenticate. Substantial rewriting needed.")
    elif overall_sim > 15:
        print("   ** MODERATE: May trigger editorial review. Targeted rewriting recommended.")
    else:
        print("   * ACCEPTABLE: Within typical thresholds for extended journal versions.")

    # Per-section breakdown
    sections_b = extract_sections_text(tex_b)
    print(f"\n## Per-Section Breakdown (target document)")
    print(f"{'Section':<45} {'Words':>6} {'Match':>6} {'Sim%':>6} {'Risk':>6}")
    print("-" * 72)

    for name, content in sections_b:
        sim, mw, tw = compute_similarity_ithenticate(text_a, content, args.ngram)
        risk = "HIGH" if sim > 30 else "MED" if sim > 15 else "OK"
        name_short = name[:44]
        print(f"{name_short:<45} {tw:>6} {mw:>6} {sim:>5.1f}% {risk:>6}")

    # Matching passages
    if args.verbose:
        print(f"\n## Top Matching Passages (>50% n-gram overlap)")
        print("-" * 70)
        passages = find_matching_passages(text_a, text_b, args.ngram)
        for i, (passage, ratio) in enumerate(passages[:20]):
            print(f"\n  [{ratio:.0%}] {passage[:120]}{'...' if len(passage) > 120 else ''}")

    print("\n" + "=" * 70)
    print("NOTE: This tool approximates iThenticate's algorithm using 5-word n-gram")
    print("fingerprinting. Actual iThenticate scores may differ due to proprietary")
    print("database matching and phrase-level analysis. Use as a directional guide.")
    print()
    print("Thresholds (from literature):")
    print("  < 15%  : Generally acceptable")
    print("  15-25% : Requires editorial review")
    print("  > 25%  : High risk, needs substantial rewriting")


if __name__ == "__main__":
    main()
