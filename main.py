from data_loader import NLPDatasetLoader
from models.sarcasm_model import SarcasmClassifier
from models.emotion_classifier import EmotionClassifier
from sklearn.metrics import classification_report
def main():
    loader = NLPDatasetLoader()

    # # Load sarcasm
    sarcasm_df = loader.load_sarcasm_data()
    sarcasm_df = loader.preprocess_dataframe(sarcasm_df, text_column="comment")
    sarcasm_X_train, sarcasm_X_test, sarcasm_y_train, sarcasm_y_test = loader.get_train_test_split(sarcasm_df, "comment", "label")

    # Load emotion
    emotion_df = loader.load_emotion_data()
    emotion_df = loader.preprocess_dataframe(emotion_df, text_column="text")
    # Train
    clf = SarcasmClassifier(embedding_type='bert')
    clf.train(sarcasm_X_train, sarcasm_y_train)

    # Predict
    y_pred = clf.predict(sarcasm_X_test)

    # Evaluate
    print(classification_report(sarcasm_y_test, y_pred))



    emotion_X_train, emotion_X_test, emotion_y_train, emotion_y_test = loader.get_train_test_split(emotion_df, "text", "emotion")
    emotion_clf = EmotionClassifier(embedding_type='bert')
    emotion_clf.train(emotion_X_train, emotion_y_train)

    sample_texts = [
        "I'm thrilled we get to refactor legacy code again.",
        "This deadline extension really made my day."
    ]
    # Predict sarcasm
    preds = emotion_clf.predict(sample_texts)
    print(preds)

    # For explanation or confidence
    probs = emotion_clf.predict_proba(sample_texts)
    print(probs)


if __name__ == "__main__":
    main()
