# Golden Retriever 🐕: In-House Retrieval-Augmented Generation (RAG) System -

__Specific Research Question:__
How can we design and optimize a RAG system to provide personalized, reference-rich outputs for data science teams, while managing in-house packages, in an offline environment?

To break it down further:
- __Customized Output with References:__ Investigate how the RAG system can leverage context-specific information to generate tailored outputs for data scientists. Explore methods to include accurate and relevant references in the generated content.
- __In-House Offline System:__ Examine the technical requirements and constraints of deploying a RAG system in an offline environment. Assess the challenges related to data storage, processing, and model updates without internet connectivity.
- __Managing In-House Packages:__ Analyze strategies for integrating and managing proprietary data science packages within the RAG system. Look into the compatibility issues, version control, and dependency management to ensure seamless operation and integration.

## Overview:

Golden Retriever is an offline Retrieval-Augmented Generation (RAG) system designed for data science teams. It provides personalized, reference-rich outputs while operating in environments without internet connectivity. This system is ideal for secure settings where data privacy is paramount.

__Version:__ 1.0
__Developed by:__ Bishal Agarwal, Mehul Jain, Indhuja Muthu Kumar, and Dinesh Sai Pappuru

## Features:

- **Offline Operation**: Fully functional RAG system that works without internet connectivity
- **Advanced Document Preprocessing**: Robust pipeline for efficient document handling and content extraction
- **State-of-the-Art Embedding & Vector Storage**: Incorporates multiple embedding models and vector storage solutions
- **Optimized RAG Pipeline**: Includes intelligent query processing and high-precision document retrieval
- **User-Friendly Interface**: Streamlit-based UI for easy interaction

## Installation:

1. Clone the repository:
   ```
   git clone https://github.com/mehulfollytobevice/golden-retriever.git
   cd golden-retriever
   ```

2. Set up a virtual environment:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Configure API key:
   Set up your GROQ API key as an environment variable:
   ```
   export GROQ_API_KEY=your_groq_api_key
   ```
   Or add it directly in the `streamlit_app.py` file:
   ```python
   os.environ["GROQ_API_KEY"] = "your_groq_api_key"
   ```

5. Prepare document paths:
   Ensure your document paths are correctly set in the `streamlit_app.py` file:
   ```python
   parent_doc_path = "/path/to/your/parent_docs"
   child_doc_path = "./path/to/your/child_docs"
   ```

## Usage:

1. Launch the Streamlit app:
   ```
   streamlit run streamlit_app.py
   ```

2. Access the web interface at the URL provided by Streamlit (usually `http://localhost:8501`).

3. Use the chat interface to ask questions and interact with the RAG system.

## System Components:

- **Document Preprocessing**: Uses BSHTMLLoader and BeautifulSoup for content extraction
- **Embedding Models**: gte-large-en-v1.5, gte-large, stella_en_400M_v5
- **Vector Storage**: Chroma, FAISS
- **Text Generation**: Utilizes the llama-3.1-70b model

## Data Sources:

The system integrates documentation from multiple sources, including:
- Pandas
- NumPy
- Scikit-learn

## Evaluation:

The system has been evaluated against existing language models, focusing on handling the most recent versions of documentation from key libraries. Detailed evaluation results are available in the project documentation.

## Future Improvements:

- Implementing chat history storage.
- Further optimizing the retrieval component.
- Adding anchored links for direct navigation to relevant document sections.
- Introducing multiple generative model selection options in the UI.

## Contact:

For more information or support, please contact  [Mehul Jain](mailto:mehul.jain.codedev@gmail.com)
