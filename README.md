# 🤱 Pregnancy RAG Chatbot

A proactive, conversational AI chatbot that uses Retrieval-Augmented Generation (RAG) to assess pregnancy-related symptoms and provide risk insights based on medical literature. Built with LlamaIndex, Groq API, and deployed with Gradio.

## 🎯 Project Overview

This chatbot proactively asks pregnant women about their symptoms, retrieves relevant medical information from authoritative sources, and provides risk assessments with actionable recommendations. The system combines vector similarity search with BM25 retrieval for comprehensive medical knowledge retrieval.

### Key Features

- **Proactive Questioning**: Asks 5 targeted symptom-related questions
- **RAG-Powered Analysis**: Retrieves information from medical knowledge base
- **Risk Assessment**: Provides Low/Medium/High risk levels with recommendations
- **Conversational Follow-up**: Answers additional questions about pregnancy health
- **Medical Literature Grounded**: Based on WHO guidelines and medical research

## 🚀 Live Demo

**Hugging Face Spaces**: [Try the live demo here](https://huggingface.co/spaces/Affanp/Pregnancy_RAG_Chatbot)

## 📁 Project Structure

```
pregnancy_rag_chatbot/
├── backend/
│   ├── utils.py                 # Index management and vector store setup
│   ├── rag_functions.py         # RAG retrieval and response generation
│   └── insert_to_vectorstore.py # Vector database rebuild utility
├── frontend/
│   └── app.py                  # Gradio web interface
├── knowledge_base/             # Medical documents (CSV, TXT, PDF)
│   ├── pregnancy_symptoms.csv
│   ├── medical_guidelines.txt
│   └── ... (your medical data files)
├── requirements.txt            # Python dependencies
├── .env                       # Environment variables (not in repo)
├── .gitignore                 # Git ignore file
└── README.md                  # This file
```

## 🛠️ Technology Stack

- **LLM**: Groq API (Llama 3.1 8B Instant)
- **Vector Database**: Pinecone
- **RAG Framework**: LlamaIndex
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)
- **Frontend**: Gradio
- **Retrieval**: Hybrid (Vector + BM25)
- **Reranking**: Cross-encoder reranking

## 📋 Prerequisites

- Python 3.8+
- Groq API account
- Pinecone account
- Git

## 🔧 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Affan-p/Pregnancy_RAG_Chatbot.git
cd Pregnancy_RAG_Chatbot
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up API Keys

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX=your_pinecone_index_name_here
```

#### Getting API Keys:

**Groq API Key:**
1. Visit [Groq Console](https://console.groq.com/)
2. Sign up/login to your account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key to your `.env` file

**Pinecone API Key:**
1. Visit [Pinecone](https://www.pinecone.io/)
2. Sign up for a free account
3. Create a new project
4. Go to API Keys in your dashboard
5. Copy your API key
6. Create a new index (dimension: 384, metric: cosine)
7. Use the index name in your `.env` file

### 5. Prepare Knowledge Base

Create a `knowledge_base` folder and add your medical documents (CSV, TXT, PDF formats supported). The system includes built-in pregnancy health knowledge.

### 6. Initialize the Vector Database

**Option A: Using the dedicated rebuild script (Recommended)**
```bash
# This will rebuild the entire index with your knowledge base
python backend/insert_to_vectorstore.py
```

**Option B: Using utils directly**
```bash
# Run this once to populate your Pinecone index
python backend/utils.py
```

The rebuild script (`insert_to_vectorstore.py`) will:
1. Delete all old vectors from Pinecone
2. Clear local storage
3. Process your CSVs with the new CSV reader
4. Rebuild and store everything fresh

### 7. Run the Application

```bash
# From the root directory
python frontend/app.py
```

The application will be available at `http://localhost:7860`

## 📚 Usage

1. **Start Conversation**: The bot will greet you and ask the first symptom question
2. **Answer Questions**: Respond to 5 proactive questions about pregnancy symptoms
3. **Get Assessment**: Receive a risk level (Low/Medium/High) with recommendations
4. **Ask Follow-ups**: Continue with any additional pregnancy-related questions
5. **Reset**: Use "reset" or click the reset button to start a new assessment

## 🔄 Vector Database Management

### Rebuilding the Index

If you need to update your knowledge base or fix indexing issues:

```bash
python backend/insert_to_vectorstore.py
```

This utility script:
- Clears existing vectors from Pinecone
- Removes local storage cache
- Reprocesses all documents in `knowledge_base/`
- Rebuilds the vector index with fresh embeddings

### Checking Index Status

```python
from backend.utils import check_index_status
check_index_status()
```

## 🏥 Medical Disclaimer

⚠️ **Important**: This AI assistant provides information based on medical literature but is NOT a substitute for professional medical advice, diagnosis, or treatment. In emergencies, call emergency services immediately.

## 🔄 System Architecture

```
User Input → Symptom Collection → RAG Retrieval → Risk Assessment → Response
                ↓
    [Vector Store] ← Medical Knowledge Base → [BM25 Index]
                ↓
    [Groq LLM] ← Retrieved Context → [Response Generation]
```

## 📊 Risk Assessment Categories

- **Low Risk**: Normal pregnancy symptoms, routine monitoring recommended
- **Medium Risk**: Concerning symptoms, contact doctor within 24 hours
- **High Risk**: Serious symptoms, immediate medical attention required

## 🚀 Deployment

### Local Development
```bash
python frontend/app.py
```

### Hugging Face Spaces
1. Upload files to HF Spaces repository
2. Ensure `requirements.txt` includes all dependencies
3. Set environment variables in HF Spaces settings
4. Run the vector database rebuild on first deployment
5. The app will auto-deploy

### Production Deployment
1. Set up your environment variables
2. Run the vector database initialization:
   ```bash
   python backend/insert_to_vectorstore.py
   ```
3. Start the application:
   ```bash
   python frontend/app.py
   ```

## 🔧 Configuration Files

### backend/utils.py
Core utilities for:
- Pinecone vector store management
- Document loading and chunking
- Index creation and rebuilding

### backend/rag_functions.py
RAG implementation including:
- Hybrid retrieval (Vector + BM25)
- Response generation with Groq API
- Document reranking and filtering

### backend/insert_to_vectorstore.py
Simple utility script for rebuilding the vector database:
```python
from utils import rebuild_index

# Rebuild the entire index
index = rebuild_index()

if index:
    print("🎉 Ready to go! Your CSVs are now properly processed")
else:
    print("❌ Something went wrong with the rebuild")
```

### frontend/app.py
Gradio interface with:
- Conversational flow management
- Risk assessment logic
- Interactive chat interface

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🆘 Troubleshooting

### Common Issues:

**"Error: Retriever not available"**
- Ensure Pinecone index is created and populated
- Run `python backend/insert_to_vectorstore.py` to rebuild the index
- Check API keys in `.env` file
- Verify internet connection

**"Groq API call failed"**
- Verify GROQ_API_KEY is correct
- Check Groq API usage limits
- Ensure you have credits in your Groq account

**"No relevant documents found"**
- Populate your `knowledge_base/` folder with medical documents
- Run the index rebuild process: `python backend/insert_to_vectorstore.py`
- Check if documents were properly processed

**Empty or corrupted index**
- Delete the `storage/` folder
- Run `python backend/insert_to_vectorstore.py` to rebuild from scratch
- Verify your Pinecone index settings (dimension: 384, metric: cosine)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ for safer pregnancies worldwide**
