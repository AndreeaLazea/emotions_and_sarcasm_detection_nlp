from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def print_classification_report(y_true, y_pred, title="Classification Report"):
    print(f"\n{title}")
    print(classification_report(y_true, y_pred))


def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_word_importance(importance_list, title="Word Importance"):
    """
    Expects list of tuples: [('word1', score1), ('word2', score2), ...]
    """
    words, scores = zip(*importance_list)
    indices = np.argsort(scores)[::-1]

    plt.figure(figsize=(8, 5))
    plt.bar([words[i] for i in indices], [scores[i] for i in indices])
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.ylabel("Score Δ on Removal")
    plt.tight_layout()
    plt.show()
