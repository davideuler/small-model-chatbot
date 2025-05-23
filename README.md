---
title: Small Model Chatbot
emoji: 😻
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 5.31.0
app_file: app.py
pinned: false
license: mit
short_description: Some small models chatbot
---
=======
# Multi-Model Tiny Chatbot

A lightweight, multi-model chat application featuring several small language models optimized for different tasks. Built with Gradio for an intuitive web interface and designed for local deployment.

## 🌟 Features

- **Multiple Model Support**: Choose from 4 specialized small language models
- **Lazy Loading**: Models are loaded only when selected, optimizing memory usage
- **Real-time Chat Interface**: Smooth conversational experience with Gradio
- **Lightweight**: All models are under 200M parameters for fast inference
- **Local Deployment**: Run entirely on your local machine

## 🤖 Available Models

### 1. SmolLM2 (135M Parameters)
- **Purpose**: General conversation and instruction following
- **Architecture**: HuggingFace SmolLM2-135M-Instruct
- **Best For**: General Q&A, creative writing, coding help
- **Language**: English

### 2. NanoLM-25M (25M Parameters)
- **Purpose**: Ultra-lightweight instruction following
- **Architecture**: Mistral-based with chat template support
- **Best For**: Quick responses, simple tasks, resource-constrained environments
- **Language**: English

### 3. NanoTranslator-S (9M Parameters)
- **Purpose**: English to Chinese translation
- **Architecture**: LLaMA-based translation model
- **Best For**: Translating English text to Chinese
- **Language**: English → Chinese

### 4. NanoTranslator-XL (78M Parameters)
- **Purpose**: Enhanced English to Chinese translation
- **Architecture**: LLaMA-based with improved accuracy
- **Best For**: High-quality English to Chinese translation
- **Language**: English → Chinese

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- 4GB+ RAM recommended
- Internet connection for initial model downloads

### Installation

1. **Run the application**
   ```bash
   uv run app.py
   ```

2. **Open your browser**
   - Navigate to `http://localhost:7860`
   - Select a model and start chatting!


## 🎯 Use Cases

### General Conversation
- Use **SmolLM2** or **NanoLM-25M** for general chat, Q&A, and assistance

### Translation Tasks
- Use **NanoTranslator-S** for quick English→Chinese translations
- Use **NanoTranslator-XL** for higher quality English→Chinese translations

### Resource-Constrained Environments
- **NanoLM-25M** (25M params) for ultra-lightweight deployment
- **NanoTranslator-S** (9M params) for minimal translation needs

## 💡 Model Performance

| Model | Parameters | Use Case | Memory Usage | Speed |
|-------|------------|----------|--------------|-------|
| SmolLM2 | 135M | General Chat | ~500MB | Fast |
| NanoLM-25M | 25M | Lightweight Chat | ~100MB | Very Fast |
| NanoTranslator-S | 9M | Quick Translation | ~50MB | Very Fast |
| NanoTranslator-XL | 78M | Quality Translation | ~300MB | Fast |



### Model Sources
- SmolLM2: `HuggingFaceTB/SmolLM2-135M-Instruct`
- NanoLM-25M: `Mxode/NanoLM-25M-Instruct-v1.1`
- NanoTranslator-S: `Mxode/NanoTranslator-S`
- NanoTranslator-XL: `Mxode/NanoTranslator-XL`

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [HuggingFace](https://huggingface.co/) for the Transformers library and model hosting
- [Mxode](https://huggingface.co/Mxode) for the Nano series models
- [Gradio](https://gradio.app/) for the amazing web interface framework

