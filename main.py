import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer

class MultiModelChat:
    def __init__(self):
        self.models = {}
    
    def ensure_model_loaded(self, model_name):
        """Lazy load a model only when needed"""
        if model_name not in self.models:
            print(f"Loading {model_name} model...")
            
            if model_name == 'SmolLM2':
                self.models['SmolLM2'] = {
                    'tokenizer': AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct"),
                    'model': AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
                }
            elif model_name == 'FLAN-T5':
                self.models['FLAN-T5'] = {
                    'tokenizer': T5Tokenizer.from_pretrained("google/flan-t5-small"),
                    'model': T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
                }
            
            # Set pad token for the newly loaded model
            if self.models[model_name]['tokenizer'].pad_token is None:
                self.models[model_name]['tokenizer'].pad_token = self.models[model_name]['tokenizer'].eos_token
            
            print(f"{model_name} model loaded successfully!")
    
    def chat(self, message, history, model_choice):
        if model_choice == "SmolLM2":
            return self.chat_smol(message, history)
        elif model_choice == "FLAN-T5":
            return self.chat_flan(message, history)
    
    def chat_smol(self, message, history):
        self.ensure_model_loaded('SmolLM2')
        
        tokenizer = self.models['SmolLM2']['tokenizer']
        model = self.models['SmolLM2']['model']
        
        inputs = tokenizer(f"User: {message}\nAssistant:", return_tensors="pt")
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=80,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Assistant:")[-1].strip()
    
    def chat_flan(self, message, history):
        self.ensure_model_loaded('FLAN-T5')
        
        tokenizer = self.models['FLAN-T5']['tokenizer']
        model = self.models['FLAN-T5']['model']
        
        inputs = tokenizer(f"Answer the question: {message}", return_tensors="pt")
        outputs = model.generate(inputs.input_ids, max_length=100)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

chat_app = MultiModelChat()

def respond(message, history, model_choice):
    return chat_app.chat(message, history, model_choice)

with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# Multi-Model Tiny Chatbot")
    
    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=["SmolLM2", "FLAN-T5"],
            value="SmolLM2",
            label="Select Model"
        )
    
    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(label="Message", placeholder="Type your message here...")
    clear = gr.Button("Clear")
    
    def user_message(message, history):
        return "", history + [[message, None]]
    
    def bot_message(history, model_choice):
        user_msg = history[-1][0]
        bot_response = chat_app.chat(user_msg, history[:-1], model_choice)
        history[-1][1] = bot_response
        return history
    
    msg.submit(user_message, [msg, chatbot], [msg, chatbot]).then(
        bot_message, [chatbot, model_dropdown], chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()