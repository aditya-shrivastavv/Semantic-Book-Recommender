# Semantic Book Recommender

## Introduction

Semantic Book Recommender is a machine learning-powered application designed to provide personalized book recommendations. By leveraging advanced natural language processing techniques, the system analyzes book descriptions, tags, and user sentiment to deliver tailored suggestions. The project integrates multiple workflows, including data preprocessing, vector search, text classification, and sentiment analysis, culminating in an interactive Gradio-based dashboard for end-users.

This project is ideal for exploring the intersection of AI, semantic search, and user experience in the domain of book recommendations.

## Project Workflow

1. Data Exploration (`data-exploration.ipynb`)
   - **Output**: `books_cleaned.csv`  

2. Vector Search (`vector-search.ipynb`)
   - **Output:** `tagged_description.csv`  

3. Text Classification (`text-classification.ipynb`)
   - **Output:** `books_with_categories.csv`  

4. Sentiment Analysis (`sentiment_analysis.ipynb`)
   - **Output:** `books_with_emotions.csv`  

5. Gradio Dashboard (`gradio-dashboard.py`)
   - **Final Output:** Interactive dashboard for recommendations  

---

## Required Files and Folders

1. **`chroma_db_books_vector_database/`** — Pre-built vector database for book embeddings  
2. **`.env`** — Environment variables configuration file  
3. **`cover-not-found.jpg`** — Placeholder image for missing book covers  
4. **`gradio-dashboard.py`** — Main application file  
5. **`requirements.txt`** — List of required Python packages  
6. **`.venv/`** — Python virtual environment directory  

---

## Deployment Guide

### Python Environment Setup

- **Required Python version:** `3.11.9`  
  *Ensure the exact version for compatibility.*

1. **Install Python 3.11.9**  
   Download and install Python from [python.org](https://www.python.org/downloads/release/python-3119/).  

2. **Create a Virtual Environment:**  

   ```bash
   python3.11 -m venv .venv
   ```

3. **Activate the Virtual Environment:**  

   - **Windows (PowerShell):**

     ```bash
     .\.venv\Scripts\activate.ps1
     ```

   - **Linux/Mac:**

     ```bash
     source .venv/bin/activate
     ```

4. **Install Dependencies:**  

   ```bash
   pip install -r requirements.txt
   ```

---

### Running the Application

1. Ensure all required files are present.
2. Launch the Gradio dashboard with:  

   ```bash
   python gradio-dashboard.py
   ```

## File Explanations

### 1. Data Exploration (`data-exploration.ipynb`)

1. Downloading and saving the database from Kaggle using `kagglehub` library.
2. Converting the database to a DataFrame and exploring the data.
3. **Handling missing data**
   1. Using `seaborn` and `matplotlib` for visualization.
   2. Generating a heatmap which tells, which field has the most missing data.
4. Creating a few new columns for better analysis.
5. Creating a correlation matrix to understand the relationship between different columns.
6. Reducing the number of categories in the `categories` column. Initially we had 500+ categories, which we are planning to reduced to 5-8.
7. Then we are removing the rows which have bad data in the `description` column. We are keeping only those rows which have a description of length greater than 25.
8. Finally, we are saving the cleaned data to a new CSV file.

### 2. Vector Search (`vector-search.ipynb`)

1. Importing essential modules from `langchain`, `dotenv`, and `pandas` to handle text processing, embeddings, and environment variables.
2. Loading Environment Variables: `load_dotenv()` loads the API key (e.g., Google Generative AI key) from the `.env` file to ensure sensitive data isn’t hardcoded in the script.
3. The cleaned book dataset (`books_cleaned.csv`) is imported into a DataFrame using `pandas`. This data was prepared in the previous `data-exploration.ipynb` step.
4. **Extracting Descriptions**
   1. The `tagged_description` column (likely containing book summaries) is saved into a new file `tagged_description.txt`.
   2. Each description is written on a new line without headers or indexes — simplifying further text processing.
5. The `TextLoader()` function loads `tagged_description.txt` as a raw document, preparing it for splitting and embedding.
6. **Splitting Descriptions into Chunks**
   1. Using `CharacterTextSplitter()`, the descriptions are broken into individual documents based on newlines (`\n`).
   2. Chunk size and overlap are set to `0` to ensure each line remains a standalone document without merging content.
7. **Generating Embeddings**
   1. The script uses `GoogleGenerativeAIEmbeddings` (specifically `model="models/embedding-001"`) to generate embeddings — transforming each description into a vector representation that captures its meaning and context.
   2. These embeddings are stored in a `Chroma` vector database (`db_books`), which supports fast similarity searches.
8. **Performing a Semantic Search Query**
   1. An example query — `"A book to teach children about nature"` — is run through the vector database using `db_books.similarity_search()`.
   2. It returns the top 10 most semantically similar book descriptions (`k=10`).
   3. The result is a list of matching descriptions, and the script extracts the `isbn13` identifier from the first match to fetch the full book details from the DataFrame.
9. `Building a Recommendation Function`
   1. A function `retrieve_semantic_recommendations()` is defined to handle custom queries.
   2. It:
   3. Takes a query string (e.g., "Aliens") and number of results (`top_k`).
   4. Performs similarity search for the top 50 closest matches.
   5. Extracts book ISBNs from the search results.
   6. Filters the original DataFrame to return the top 10 recommended books based on the ISBNs.

### 3. Text Classification (`text-classification.ipynb`)

### 4. Sentiment Analysis (`sentiment_analysis.ipynb`)

### 5. Gradio Dashboard (`gradio-dashboard.py`)
