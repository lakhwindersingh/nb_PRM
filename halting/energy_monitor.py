import math
import re
from collections import Counter
from typing import Set, List


class EnergyMonitor:
    def __init__(self, threshold=0.85, decay_rate=0.05):
        """
        :param threshold: minimum confidence or novelty score to continue reasoning
        :param decay_rate: minimum delta of reasoning improvement to justify continuing
        """
        self.threshold = threshold
        self.decay_rate = decay_rate
        self.previous_scores = []
        self.seen_outputs: Set[str] = set()
        self.vocabulary: Set[str] = set()

    def compute_energy(self, new_output: str) -> float:
        """
        Optimized energy computation based on multiple factors:
        - Semantic complexity (vocabulary diversity)
        - Information density (entropy)
        - Novelty (uniqueness vs previous outputs)
        - Coherence (structural quality)
        - Computational efficiency (length penalty for diminishing returns)

        Returns a score between 0 and 1, where higher values indicate
        better energy efficiency and reasoning quality.
        """
        if not new_output or not new_output.strip():
            return 0.0

        # Normalize input
        text = new_output.strip().lower()
        words = re.findall(r'\b\w+\b', text)

        if not words:
            return 0.0

        # 1. Semantic Complexity Score (0-1)
        unique_words = set(words)
        complexity_score = len(unique_words) / max(len(words), 1)

        # Update global vocabulary for novelty calculation
        self.vocabulary.update(unique_words)

        # 2. Information Density (Entropy) Score (0-1)
        word_freq = Counter(words)
        entropy = 0.0
        total_words = len(words)

        for count in word_freq.values():
            prob = count / total_words
            entropy -= prob * math.log2(prob) if prob > 0 else 0

        # Normalize entropy (max possible entropy is log2(unique_words))
        max_entropy = math.log2(len(unique_words)) if len(unique_words) > 1 else 1
        entropy_score = entropy / max_entropy if max_entropy > 0 else 0

        # 3. Novelty Score (0-1)
        # Check similarity with previous outputs
        novelty_score = 1.0
        text_hash = hash(text)

        if text in self.seen_outputs:
            novelty_score = 0.0  # Exact duplicate
        else:
            # Calculate Jaccard similarity with previous outputs
            current_words = set(words)
            max_similarity = 0.0

            for prev_output in list(self.seen_outputs)[-5:]:  # Check last 5 outputs
                prev_words = set(re.findall(r'\b\w+\b', prev_output))
                if prev_words:
                    intersection = len(current_words.intersection(prev_words))
                    union = len(current_words.union(prev_words))
                    similarity = intersection / union if union > 0 else 0
                    max_similarity = max(max_similarity, similarity)

            novelty_score = 1.0 - max_similarity

        self.seen_outputs.add(text)

        # Keep only recent outputs to prevent memory bloat
        if len(self.seen_outputs) > 100:
            self.seen_outputs = set(list(self.seen_outputs)[-50:])

        # 4. Coherence Score (0-1)
        # Simple coherence based on sentence structure and punctuation
        sentences = re.split(r'[.!?]+', text)
        valid_sentences = [s.strip() for s in sentences if s.strip()]

        coherence_score = 0.5  # baseline
        if valid_sentences:
            # Prefer outputs with proper sentence structure
            avg_sentence_length = sum(len(s.split()) for s in valid_sentences) / len(valid_sentences)
            # Optimal sentence length is around 10-20 words
            if 5 <= avg_sentence_length <= 25:
                coherence_score += 0.3

            # Bonus for proper punctuation usage
            if any(p in new_output for p in '.!?'):
                coherence_score += 0.2

        # 5. Length Efficiency Score (0-1)
        # Penalize extremely long outputs (diminishing returns)
        optimal_length = 100  # words
        length_ratio = len(words) / optimal_length

        if length_ratio <= 1.0:
            length_score = length_ratio  # Linear growth up to optimal
        else:
            # Logarithmic decay after optimal length
            length_score = 1.0 - (math.log(length_ratio) / math.log(10))

        length_score = max(0.1, min(1.0, length_score))  # Clamp between 0.1 and 1.0

        # 6. Weighted Final Score
        weights = {
            'complexity': 0.25,
            'entropy': 0.20,
            'novelty': 0.25,
            'coherence': 0.15,
            'length': 0.15
        }

        final_score = (
                weights['complexity'] * complexity_score +
                weights['entropy'] * entropy_score +
                weights['novelty'] * novelty_score +
                weights['coherence'] * coherence_score +
                weights['length'] * length_score
        )

        return round(final_score, 3)

    def should_continue(self, new_output: str) -> bool:
        score = self.compute_energy(new_output)
        self.previous_scores.append(score)

        if score < self.threshold:
            return False  # confidence/novelty too low

        if len(self.previous_scores) >= 2:
            delta = abs(self.previous_scores[-1] - self.previous_scores[-2])
            if delta < self.decay_rate:
                return False  # convergence detected

        return True

    def get_energy_breakdown(self, new_output: str) -> dict:
        """
        Returns detailed breakdown of energy computation for debugging/analysis
        """
        if not new_output or not new_output.strip():
            return {"error": "Empty input"}

        text = new_output.strip().lower()
        words = re.findall(r'\b\w+\b', text)

        if not words:
            return {"error": "No valid words found"}

        # Calculate individual components (simplified version of main function)
        unique_words = set(words)
        complexity_score = len(unique_words) / max(len(words), 1)

        word_freq = Counter(words)
        entropy = sum(-p / len(words) * math.log2(p / len(words))
                      for p in word_freq.values() if p > 0)
        max_entropy = math.log2(len(unique_words)) if len(unique_words) > 1 else 1
        entropy_score = entropy / max_entropy if max_entropy > 0 else 0

        return {
            "complexity_score": round(complexity_score, 3),
            "entropy_score": round(entropy_score, 3),
            "word_count": len(words),
            "unique_words": len(unique_words),
            "vocabulary_size": len(self.vocabulary),
            "final_score": self.compute_energy(new_output)
        }
