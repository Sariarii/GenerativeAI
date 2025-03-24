from flask import Flask, render_template, request
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the GPT-2 product description generator model and the base GPT-2 tokenizer
model = GPT2LMHeadModel.from_pretrained("HamidRezaAttar/gpt2-product-description-generator")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # Use the base GPT-2 tokenizer

# Initialize Flask app
app = Flask(__name__)

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to generate product description content
@app.route('/generate', methods=['POST'])
def generate():
    if request.method == 'POST':
        # Get the input text from the form
        input_text = request.form['input_text']
        
        # Encode the input text for the model
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        
        # Generate product description
        outputs = model.generate(input_ids, max_length=150, num_return_sequences=1, do_sample=True, top_k=50)
        
        # Decode the output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Render the result on the same page
        return render_template('index.html', input_text=input_text, generated_text=generated_text)

# Start the Flask server
if __name__ == '__main__':
    app.run(debug=True)
