import os
from datasets import Dataset, DatasetDict
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer

# Definisci un nome per il dataset
dataset_name = "Dataset_Diego"

# Esempio di dati di addestramento e valutazione
train_data = {
    'text': [
        "Questo è un esempio di frase positiva.",
        "Mi piace molto questo prodotto!",
        "Un'esperienza fantastica, lo consiglio a tutti.",
        "Questa è una frase negativa.",
        "Non mi piace affatto questo film.",
        "È stata una pessima esperienza.",
        "Ho trovato questo libro noioso.",
        "Non lo raccomando."
    ],
    'label': [1, 1, 1, 0, 0, 0, 0, 0]  # 1 per positivo, 0 per negativo
}

eval_data = {
    'text': [
        "Questo prodotto è eccezionale.",
        "Un vero disastro, mai più.",
        "Un'esperienza molto piacevole.",
        "Non vale la pena comprarlo."
    ],
    'label': [1, 0, 1, 0]
}

# Crea il dataset di addestramento e di valutazione
train_dataset = Dataset.from_dict(train_data)
eval_dataset = Dataset.from_dict(eval_data)

# Crea un DatasetDict per gestire i due dataset
dataset = DatasetDict({
    'train': train_dataset,
    'validation': eval_dataset
})

# Carica il tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Funzione per tokenizzare i dati
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Applica la tokenizzazione al dataset di addestramento e di valutazione
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Rimuovi le colonne non necessarie
tokenized_datasets = tokenized_datasets.remove_columns(["text"])

# Carica il modello
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Configura gli argomenti di training
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=10,
    evaluation_strategy="steps",
    fp16=True,  # Attiva FP16 se stai usando mixed precision
)

# Crea il trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Avvia il fine-tuning
trainer.train()

# Salva il modello e il tokenizer
save_directory = os.path.expanduser("~/Progetti/Fine_Tuning/Modello")  # Specifica il percorso di salvataggio
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

# Salva il dataset di addestramento e di valutazione su disco
train_dataset.save_to_disk(os.path.expanduser(f"~/Progetti/Fine_Tuning/Dataset/{dataset_name}/train"))
eval_dataset.save_to_disk(os.path.expanduser(f"~/Progetti/Fine_Tuning/Dataset/{dataset_name}/eval"))

print(f"Modello e tokenizer salvati con successo in: {save_directory}")
print(f"Dataset '{dataset_name}' salvati con successo.")
