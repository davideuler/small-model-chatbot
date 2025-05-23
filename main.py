import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

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
            elif model_name == 'NanoLM-25M':
                self.models['NanoLM-25M'] = {
                    'tokenizer': AutoTokenizer.from_pretrained("Mxode/NanoLM-25M-Instruct-v1.1"),
                    'model': AutoModelForCausalLM.from_pretrained("Mxode/NanoLM-25M-Instruct-v1.1")
                }
            elif model_name == 'NanoTranslator-S':
                self.models['NanoTranslator-S'] = {
                    'tokenizer': AutoTokenizer.from_pretrained("Mxode/NanoTranslator-S"),
                    'model': AutoModelForCausalLM.from_pretrained("Mxode/NanoTranslator-S")
                }
            elif model_name == 'NanoTranslator-XL':
                self.models['NanoTranslator-XL'] = {
                    'tokenizer': AutoTokenizer.from_pretrained("Mxode/NanoTranslator-XL"),
                    'model': AutoModelForCausalLM.from_pretrained("Mxode/NanoTranslator-XL")
                }
            
            # Set pad token for the newly loaded model
            if self.models[model_name]['tokenizer'].pad_token is None:
                self.models[model_name]['tokenizer'].pad_token = self.models[model_name]['tokenizer'].eos_token
            
            print(f"{model_name} model loaded successfully!")
    
    def chat(self, message, history, model_choice):
        if model_choice == "SmolLM2":
            return self.chat_smol(message, history)
        elif model_choice == "NanoLM-25M":
            return self.chat_nanolm(message, history)
        elif model_choice == "NanoTranslator-S":
            return self.chat_translator(message, history)
        elif model_choice == "NanoTranslator-XL":
            return self.chat_translator_xl(message, history)
    
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
    
    def chat_nanolm(self, message, history):
        self.ensure_model_loaded('NanoLM-25M')
        
        tokenizer = self.models['NanoLM-25M']['tokenizer']
        model = self.models['NanoLM-25M']['model']
        
        # Use chat template for NanoLM
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt")
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    
    def chat_translator(self, message, history):
        self.ensure_model_loaded('NanoTranslator-S')
        
        tokenizer = self.models['NanoTranslator-S']['tokenizer']
        model = self.models['NanoTranslator-S']['model']
        
        # Use translation prompt format
        prompt = f"<|im_start|>{message}<|endoftext|>"
        inputs = tokenizer([prompt], return_tensors="pt")
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            temperature=0.55,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    
    def chat_translator_xl(self, message, history):
        self.ensure_model_loaded('NanoTranslator-XL')
        
        tokenizer = self.models['NanoTranslator-XL']['tokenizer']
        model = self.models['NanoTranslator-XL']['model']
        
        # Use translation prompt format
        prompt = f"<|im_start|>{message}<|endoftext|>"
        inputs = tokenizer([prompt], return_tensors="pt")
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            temperature=0.55,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

chat_app = MultiModelChat()

def respond(message, history, model_choice):
    return chat_app.chat(message, history, model_choice)

with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# 🤖 Multi-Model Tiny Chatbot")
    gr.Markdown("*Lightweight AI models for different tasks - Choose the right model for your needs!*")
    
    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=["SmolLM2", "NanoLM-25M", "NanoTranslator-S", "NanoTranslator-XL"],
            value="NanoLM-25M",
            label="Select Model",
            info="Choose the best model for your task"
        )
    
    # Model information display
    with gr.Row():
        model_info = gr.Markdown(
            """
            ## 📋 NanoLM-25M (25M) - Selected
            **Best for:** Quick responses, simple tasks, resource-constrained environments  
            **Language:** English  
            **Memory:** ~100MB  
            **Speed:** Very Fast  
            
            💡 **Tip:** Ultra-lightweight model perfect for fast responses!
            """,
            visible=True
        )
    
    chatbot = gr.Chatbot(height=400, show_label=False)
    msg = gr.Textbox(
        label="Message", 
        placeholder="Type your message here...",
        lines=2
    )
    
    with gr.Row():
        clear = gr.Button("🗑️ Clear Chat", variant="secondary")
        submit = gr.Button("💬 Send", variant="primary")
    
    # Usage tips
    with gr.Accordion("📖 Model Usage Guide", open=False):
        gr.Markdown("""
        ### 🎯 When to use each model:
        
        **🔵 SmolLM2 (135M)**
        - General conversations and questions
        - Creative writing tasks
        - Coding help and explanations
        - Educational content
        
        **🟢 NanoLM-25M (25M)** 
        - Quick responses when speed matters
        - Resource-constrained environments
        - Simple Q&A tasks
        - Mobile or edge deployment
        
        **🔴 NanoTranslator-S (9M)**
        - Fast English → Chinese translation
        - Basic translation needs
        - Ultra-low memory usage
        - Real-time translation
        
        **🟡 NanoTranslator-XL (78M)**
        - High-quality English → Chinese translation
        - Professional translation work
        - Complex sentences and idioms
        - Better context understanding
        
        ### 💡 Pro Tips:
        - Models load automatically when first selected (lazy loading)
        - Translation models work best with clear, complete sentences
        - For translation, input English text and get Chinese output
        - Restart the app to free up memory from unused models
        """)
    
    def update_model_info(model_choice):
        info_map = {
            "SmolLM2": """
            ## 📋 SmolLM2 (135M) - Selected
            **Best for:** General conversation, Q&A, creative writing, coding help  
            **Language:** English  
            **Memory:** ~500MB  
            **Speed:** Fast  
            
            💡 **Tip:** Great all-around model for most conversational tasks!
            """,
            "NanoLM-25M": """
            ## 📋 NanoLM-25M (25M) - Selected
            **Best for:** Quick responses, simple tasks, resource-constrained environments  
            **Language:** English  
            **Memory:** ~100MB  
            **Speed:** Very Fast  
            
            💡 **Tip:** Ultra-lightweight model perfect for fast responses!
            """,
            "NanoTranslator-S": """
            ## 📋 NanoTranslator-S (9M) - Selected
            **Best for:** Fast English → Chinese translation  
            **Language:** English → Chinese  
            **Memory:** ~50MB  
            **Speed:** Very Fast  
            
            💡 **Tip:** Input English text to get Chinese translation. Great for quick translations!
            """,
            "NanoTranslator-XL": """
            ## 📋 NanoTranslator-XL (78M) - Selected
            **Best for:** High-quality English → Chinese translation  
            **Language:** English → Chinese  
            **Memory:** ~300MB  
            **Speed:** Fast  
            
            💡 **Tip:** Best translation quality for complex sentences and professional use!
            """
        }
        return info_map.get(model_choice, "")
    
    # Update model info when dropdown changes
    model_dropdown.change(
        update_model_info,
        inputs=[model_dropdown],
        outputs=[model_info]
    )
    
    def user_message(message, history):
        return "", history + [[message, None]]
    
    def bot_message(history, model_choice):
        user_msg = history[-1][0]
        bot_response = chat_app.chat(user_msg, history[:-1], model_choice)
        history[-1][1] = bot_response
        return history
    
    # Handle message submission
    msg.submit(user_message, [msg, chatbot], [msg, chatbot]).then(
        bot_message, [chatbot, model_dropdown], chatbot
    )
    submit.click(user_message, [msg, chatbot], [msg, chatbot]).then(
        bot_message, [chatbot, model_dropdown], chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()