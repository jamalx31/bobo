# AI voice assistant

## Overview
This project combines audio transcription capabilities with AI-powered interaction, allowing users to transcribe audio and engage in conversations with an AI model.

## Features
- Audio transcription using Whisper
- Integration with Ollama for AI-powered conversations
- Real-time visual analysis using computer vision and AI
- Describe objects, scenes, and activities captured by webcam or screenshot

## Requirements
- Python 3.8+
- See `requirements.txt` for a full list of dependencies

## Installation
1. Clone the repository
2. Create and activate a Python virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
4. Install Ollama following the instructions at [ollama.ai](https://ollama.ai)

## Usage
1. Ensure your virtual environment is activated
2. Run the main script:
   ```
   python bobo.py
   ```
3. Ask Bobo to tell you what they see or look at your screen!

## Configuration
- Customize AI model prompts as needed

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License

## Acknowledgements
- OpenAI Whisper
- Faster Whisper
- Ollama
- All other open-source libraries used in this project


