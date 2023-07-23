# Import necessary libraries
import gradio as gr 
from transformers import AutoTokenizer
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

# Load the pre-trained tokenizer and model from the Helsinki-NLP/opus-mt-en-hi model
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")

# Load the pre-trained model for causal language modeling with specific configurations
model = TFAutoModelForSeq2SeqLM.from_pretrained("tf_model1/")

# Function to translate the input text using the pre-trained model
def translate(text):
    tokenized = tokenizer([text], return_tensors='tf')
    out = model.generate(**tokenized, max_length=128)
    translated_text = tokenizer.decode(out[0], skip_special_tokens=True)
    return translated_text

# Define the Gradio interface
iface = gr.Interface(
    fn=translate,
    inputs=gr.Textbox(lines=20, label="Text"),
    outputs=gr.Textbox(label="Translation", lines=20),
    live=True
)

# Launch the Gradio interface
iface.launch()