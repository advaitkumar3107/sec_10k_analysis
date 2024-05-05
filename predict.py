import torch
import scipy

def prediction(model, tokenizer, X):
    preds = []
    preds_proba = []
    tokenizer_kwargs = {"padding": True, "truncation": True, "max_length": 512}
    for x in X:
        with torch.no_grad():
            input_sequence = tokenizer(x, return_tensors="pt", **tokenizer_kwargs)
            logits = model(**input_sequence).logits
            scores = {
            k: v
            for k, v in zip(
                model.config.id2label.values(),
                scipy.special.softmax(logits.numpy().squeeze()),
            )
        }
        sentimentFinbert = max(scores, key=scores.get)
        probabilityFinbert = max(scores.values())
        preds.append(sentimentFinbert)
        preds_proba.append(probabilityFinbert)
    return preds, preds_proba