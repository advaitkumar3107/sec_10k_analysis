import torch
import scipy

def prediction(model, tokenizer, X):
    """
    Input : 
    model (AutoModelForSequenceClassification/BertModelForSequenceClassification) : The NLP model to be used for prediction
    tokenizer (AutoTokenizer/BertTokenizer) : The tokenizer to convert the sentence into corresponding tokens to feed into the model
    X (list[str]) : List of sentences on which the prediction needs to be done
    
    Output : 
    preds (list[str]) : List of the predicted outputs for each sentence
    preds_proba (list[float]) : Prob. of prediction being correct
    """
    
    preds = []
    preds_proba = []
    tokenizer_kwargs = {"padding": True, "truncation": True, "max_length": 512}  ## initialise tokenizer params
    for x in X:
        with torch.no_grad():  ## So that the model weights remain fixed
            input_sequence = tokenizer(x, return_tensors="pt", **tokenizer_kwargs)   ## convert sentence into corresponding torch.Tensor
            logits = model(**input_sequence).logits   ## feed the input and get the corresponding logits

        #### Calculate scores for each label using softmax and map it to the correct label according to the model config file
            scores = {
            k: v
            for k, v in zip(
                model.config.id2label.values(),
                scipy.special.softmax(logits.numpy().squeeze()),
            )
        }
        sentimentFinbert = max(scores, key=scores.get)  ## Get the label that has the maximum score
        probabilityFinbert = max(scores.values())   ## Get the prob of that label
        preds.append(sentimentFinbert)   ## append the label 
        preds_proba.append(probabilityFinbert)   ## append the prob
    return preds, preds_proba