import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Modèle de base et modèle PEFT entraîné
base_model = "meta-llama/Llama-3.2-1B"
peft_model_id = "Sariari/autotrain-63uch-0pcuy"

# Chargement du tokenizer et correction du pad_token
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token  # Correction pour éviter le conflit pad/eos

# Chargement du modèle sur CPU (ou auto si GPU dispo)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float32,  # Float32 pour stabilité sur CPU
    device_map="cpu"            # Utilisation CPU
)

# Chargement du modèle PEFT
model = PeftModel.from_pretrained(model, peft_model_id).eval()

# Définition du prompt
prompt = "Question : Quelle est la garantie de vos produits ? Réponse :"

# Tokenisation avec gestion du attention_mask
inputs = tokenizer(
    prompt, 
    return_tensors='pt', 
    padding=True, 
    truncation=True
)
input_ids = inputs.input_ids.to(model.device)
attention_mask = inputs.attention_mask.to(model.device)

# Génération avec paramètres optimisés
output_ids = model.generate(
    input_ids,
    attention_mask=attention_mask,  # Correction : ajout du mask
    max_new_tokens=50,
    pad_token_id=tokenizer.eos_token_id,
    temperature=0.1,  # Réduit la variabilité pour des réponses plus précises
    top_p=0.9,  # Exclut les tokens trop improbables
)

# Conversion des logits en float32 (prévention d'erreur)
logits = model(input_ids).logits.float()

# Décodage de la réponse
response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

# Affichage de la réponse générée
print("Réponse du modèle :", response)
