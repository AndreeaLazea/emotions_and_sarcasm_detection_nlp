def compare_embeddings(models, texts, class_index=1):
    """
    Args:
        models: list of (name, model) tuples
        texts: list of sample strings
    """
    for name, model in models:
        print(f"\n==={name}===")
        for text in texts:
            proba = model.predict_proba([text])[0][class_index]
            print(f"{text[:40]}... → class {class_index} prob: {proba:.4f}")
