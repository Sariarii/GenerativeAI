from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Définition du device : GPU si disponible, sinon CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Chargement du modèle et du tokenizer, et déplacement du modèle sur le device 
#pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B").to(device)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# Initialisation de l'application Flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    if request.method == 'POST':
        # Récupération du texte saisi
        input_text = request.form['input_text']
        # Encodage du texte d'entrée et déplacement sur le device
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        
        # Génération du texte avec des paramètres plus précis
        outputs = model.generate(
            input_ids,
            max_length=150,
            num_return_sequences=1,
            do_sample=True,         # Active l'échantillonnage
            top_k=50,               # Garde les 50 tokens les plus probables
            top_p=0.95,             # Nucleus sampling : conserve les tokens cumulant 95% de probabilité
            temperature=0.3,        # Contrôle l'aléatoire de la génération (plus bas = plus déterministe)
            repetition_penalty=1.2, # Pénalise les répétitions pour éviter les redondances
            early_stopping=True     # Arrête la génération lorsque la séquence est terminée
        )
        
        # Décodage de la séquence générée
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Affichage du résultat dans la page HTML
        return render_template('index.html', input_text=input_text, generated_text=generated_text)

if __name__ == '__main__':
    app.run(debug=True)
