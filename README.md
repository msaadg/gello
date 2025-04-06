# Gello:

Gello is a chatbot that integrates LangChain with Google Gemini API to process text, images, and PDFs, enabling interactive question-answering and multimodal conversations.

## Features

- **Text Chat**: Interact with the AI using text inputs.  
- **Image Processing**: Upload images and ask related questions.  
- **PDF Reader**: Upload PDFs and query their content.

## Setup

### Prerequisites

- Python 3.8+  
- Streamlit  
- LangChain  
- Google Gemini API Key

### Installation

1. **Clone the Repository**  
   ```sh
   git clone https://github.com/msaadg/gello.git
   cd gello
   ```

2. **Set Up Virtual Environment**  
   ```sh
   python -m venv env
   source env/bin/activate  # On Windows: `env\Scripts\activate`
   ```

3. **Install Dependencies**  
   ```sh
   pip install -r requirements.txt
   ```

4. **Add API Key**  
   Create a `.env` file and add your Gemini API key:  
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

## Usage

Run the app:  
```sh
streamlit run app.py
```

### Navigation

- **Sidebar**: Switch between functionalities:  
  - `Text Chat`: Enter text to chat with the AI.  
  - `Image Processing`: Upload an image and ask questions.  
  - `PDF Reader`: Upload a PDF and query its content.

## Contributing

1. Fork the repo.  
2. Create a branch (`git checkout -b feature-branch`).  
3. Commit changes (`git commit -m 'Add feature'`).  
4. Push (`git push origin feature-branch`).  
5. Open a pull request.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contact

Developed by Muhammad Saad  
- LinkedIn: [Muhammad Saad](https://www.linkedin.com/in/msaad01)  
- GitHub: [msaadg](https://github.com/msaadg)
