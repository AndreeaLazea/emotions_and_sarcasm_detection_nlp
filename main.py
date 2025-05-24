from data_loader import NLPDatasetLoader
from models.sarcasm_model import SarcasmClassifier
from models.emotion_classifier import EmotionClassifier
from explainer.perturbation_explainer import PerturbationExplainer
from analysis.embedding_comparison import compare_embeddings
from evaluate import print_classification_report, plot_confusion_matrix, plot_word_importance
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

def main():
    loader = NLPDatasetLoader()

    # Load and preprocess sarcasm data
    sarcasm_df = loader.load_sarcasm_data()
    sarcasm_df = loader.preprocess_dataframe(sarcasm_df, text_column="comment")
    sarcasm_X_train, sarcasm_X_test, sarcasm_y_train, sarcasm_y_test = loader.get_train_test_split(
        sarcasm_df, "comment", "label"
    )

    # Train sarcasm classifier
    sarcasm_clf = SarcasmClassifier(embedding_type='bert')
    sarcasm_clf.train(sarcasm_X_train, sarcasm_y_train)
    sarcasm_preds = sarcasm_clf.predict(sarcasm_X_test)

    print_classification_report(sarcasm_y_test, sarcasm_preds)
    plot_confusion_matrix(sarcasm_y_test, sarcasm_preds, labels=[0, 1])

    # Load and preprocess emotion data
    emotion_df = loader.load_emotion_data()
    emotion_df = loader.preprocess_dataframe(emotion_df, text_column="text")

    # Pick ONE emotion column to classify (multi-class example)
    emotion_target = "joy"  # <-- you can change this to 'anger', 'sadness', etc.
    emotion_df = emotion_df[["text", emotion_target]].dropna()
    emotion_df[emotion_target] = emotion_df[emotion_target].astype(int)

    X_train, X_test, y_train, y_test = loader.get_train_test_split(
        emotion_df, "text", emotion_target
    )

    # Train emotion classifiers
    glove_model = EmotionClassifier(embedding_type='glove')
    glove_model.train(X_train, y_train)

    bert_model = EmotionClassifier(embedding_type='bert')
    bert_model.train(X_train, y_train)

    # Predict and evaluate
    y_pred_bert = bert_model.predict(X_test)
    print_classification_report(y_test, y_pred_bert)
    plot_confusion_matrix(y_test, y_pred_bert)

    # Explain sarcasm prediction
    explainer = PerturbationExplainer(model=sarcasm_clf)
    word_scores = explainer.explain("Oh great, another email thread", class_index=1)
    plot_word_importance(word_scores, title="Sarcasm Word Importance")

    # Compare embeddings
    texts = [
        "Oh wow, what a fantastic idea.",
        "Sure, I love getting feedback at 2am.",
        "I'm genuinely happy with the results."
    ]
    compare_embeddings(
        models=[("GloVe", glove_model), ("BERT", bert_model)],
        texts=texts,
        class_index=1
    )

if __name__ == "__main__":
    main()
