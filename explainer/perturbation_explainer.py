import numpy as np

class PerturbationExplainer:
    def __init__(self, model):
        self.model = model

    def explain(self, text, class_index=1):
        """
        Args:
            text (str): Input sentence
            class_index (int): Index of the class you want to explain (e.g., 1 for sarcastic)
        Returns:
            list of (word, confidence_change)
        """
        original_proba = self.model.predict_proba([text])[0][class_index]
        words = text.split()
        importance = []

        for i in range(len(words)):
            perturbed = words[:i] + words[i+1:]
            perturbed_text = ' '.join(perturbed)
            new_proba = self.model.predict_proba([perturbed_text])[0][class_index]
            delta = original_proba - new_proba
            importance.append((words[i], delta))

        return sorted(importance, key=lambda x: -abs(x[1]))
