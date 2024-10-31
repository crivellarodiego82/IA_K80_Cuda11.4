from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Carica il modello e il tokenizer salvati
model_path = "/home/nextserver/Progetti/Fine_Tuning/Modello"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Funzione di inferenza
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = logits.argmax().item()
    return "Positiva" if predicted_class == 1 else "Negativa"

# Esempi di utilizzo
print(predict_sentiment("Questo prodotto è fantastico!"))
print(predict_sentiment("Non comprerò mai più questo servizio."))
