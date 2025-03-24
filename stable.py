import torch
from diffusers import StableDiffusionPipeline
import gradio as gr

# Charger le modèle sur le CPU en float32
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float32
)
pipe = pipe.to("cpu")

def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

# Créer l'interface Gradio avec un champ de texte et une sortie image
iface = gr.Interface(fn=generate_image, inputs="text", outputs="image")

# Lancer l'interface web (affichera un lien localhost)
iface.launch()