# halting/energy_monitor.py

class EnergyMonitor:
    def __init__(self, threshold=0.85, decay_rate=0.05):
        """
        :param threshold: minimum confidence or novelty score to continue reasoning
        :param decay_rate: minimum delta of reasoning improvement to justify continuing
        """
        self.threshold = threshold
        self.decay_rate = decay_rate
        self.previous_scores = []

    def compute_energy(self, new_output):
        """
        Dummy scoring function. Replace with embedding similarity, entropy, etc.
        """
        score = len(set(new_output.lower().split())) / max(len(new_output.split()), 1)
        return round(score, 3)

    def should_continue(self, new_output):
        score = self.compute_energy(new_output)
        self.previous_scores.append(score)

        if score < self.threshold:
            return False  # confidence/novelty too low

        if len(self.previous_scores) >= 2:
            delta = abs(self.previous_scores[-1] - self.previous_scores[-2])
            if delta < self.decay_rate:
                return False  # convergence detected

        return True
