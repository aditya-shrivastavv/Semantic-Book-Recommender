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

> ***Objective***: Explore and clean the book dataset (`books_with_categories.csv`) — analyze categories, descriptions, and other features to prepare data for sentiment analysis and recommendation modeling. Includes visualizations, statistics, and preprocessing steps for better downstream performance.

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

> ***Objective***: Build a semantic vector search system using book descriptions. It converts text into embeddings with Google Generative AI, stores them in a Chroma vector database, and supports similarity-based retrieval to find books that match user queries.

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

> ***Objective***: Perform sentiment and emotion analysis on book descriptions using a transformer-based model (`j-hartmann/emotion-english-distilroberta-base`). Extract dominant emotions like joy, sadness, fear, etc., to enrich book data with emotional scores for personalized recommendations.

1. **Loading the Cleaned Dataset**
   1. The cleaned book data (`books_cleaned.csv`) is imported using `pandas`.
   2. Initial category counts are displayed with `value_counts()` to understand the distribution of genres.
2. **Defining a Category Mapping**
   1. A dictionary (`category_mapping`) is created to merge multiple genres into broader, simpler categories — like converting `"Juvenile Fiction"` to `"Children’s Fiction"` and `"History"` to `"Nonfiction"`.
   2. A new column `simple_categories` is created by applying this mapping to the `categories` column.
   3. Rows with undefined categories (`NaN`) are filtered out for now.
   4. ✅ Result: Books now have cleaner, more manageable categories.
3. **Setting Up the Classification Model**
   1. The `transformers` library is used to load a zero-shot classification pipeline (`facebook/bart-large-mnli`).
   2. This allows us to classify descriptions into categories without needing a predefined training dataset.
4. **Testing the Model with a Sample Description**
   1. A random book from the `"Fiction"` category is picked, and the model predicts whether it’s `"Fiction"` or `"Nonfiction"`.
   2. The model outputs a list of scores per category, and the highest score determines the predicted category.
5. **Defining the Prediction Function**
   1. A function `generate_predictions()` is created to:
      1. Take a book description (`sequence`) and a list of possible categories.
      2. Pass the description through the model’s pipeline.
      3. Return the category with the highest confidence score.
   2. ✅ Result: Reusable function to classify any book’s description.
6. **Evaluating Model Accuracy**
   1. To measure the model's performance:
      1. 300 Fiction books and 300 Nonfiction books are classified.
      2. Actual and predicted categories are saved into `actual_cats` and `predicted_cats` lists.
   2. A DataFrame (`predictions_df`) compares the actual vs. predicted categories.
   3. It calculates how many predictions are correct using `np.where()`, outputting the accuracy percentage.
   4. ✅ Goal: Ensure the model performs well before handling missing categories.
7. **Handling Missing Categories**
   1. Books with missing `simple_categories` are identified and extracted into a `missing_cats` DataFrame (containing `isbn13` and `description`).
   2. Each missing book description is classified using the `generate_predictions()` function, and predictions are stored with corresponding ISBNs.
8. **Merging Predicted Categories Back to the Dataset**
   1. The predicted categories are merged back into the main `books` DataFrame based on `isbn13`.
   2. Missing categories are replaced with the new predictions.
   3. The temporary predicted_categories column is dropped after merging.
   4. ✅ Result: Every book now has a `simple_category` — no more missing data!
9. **Saving the Final Dataset**
   1. The final dataset, now enriched with consistent categories, is saved to `books_with_categories.csv` for further analysis.

### 4. Sentiment Analysis (`sentiment_analysis.ipynb`)

> ***Objective***: Analyze book descriptions to detect emotional tones (`joy`, `sadness`, `fear`, etc.) using a pre-trained Hugging Face emotion classifier, then save results as a new CSV (`books_with_emotions.csv`) for further recommendation analysis.

1. **Loading the Enhanced Dataset**
   1. The dataset from the previous step (`books_with_categories.csv`) is imported using `pandas`.
   2. Each book now has a `simple_category` — and this file adds emotional analysis.
2. **Setting Up the Sentiment Model**
   1. The `transformers` library loads an emotion classification model (`j-hartmann/emotion-english-distilroberta-base`).
   2. This model supports 7 emotions (anger, disgust, fear, joy, sadness, surprise, neutral) and outputs scores for each.
   3. Initial tests run on a simple string ("I love this!") and the first book’s description — split into sentences for granular analysis.
   4. ✅ Result: The model works and returns detailed emotion probabilities.
3. **Defining Emotion Scoring Logic**
   1. `calculate_max_emotion_scores()` is created to:
      1. Take model predictions (per sentence).
      2. Extract scores for each of the 7 emotions.
      3. Return the maximum score per emotion (i.e., the strongest emotional presence across all sentences).
   2. ✅ Goal: Capture the most intense expression of each emotion per book.
4. **Processing All Books**
   1. The script loops over all book descriptions (`tqdm` adds a progress bar).
   2. Each description is split into sentences and passed through the model.
   3. The max scores per emotion are calculated and stored into a new DataFrame (`emotions_df`), alongside each book’s ISBN.
   4. ✅ Result: Each book now has a comprehensive emotional fingerprint.
5. **Merging Emotions into the Main Dataset**
   1. The `emotions_df` DataFrame is merged into the main `books` DataFrame using `isbn13` as the key.
   2. The result is a book dataset enriched with emotional data.
6. **Saving the Final Dataset**
   1. The fully enriched dataset is saved as `books_with_emotions.csv` — now complete with categories and emotional analysis.
   2. ✅ Final Output: Books now have:
      1. Simplified Categories (Fiction, Nonfiction)
      2. Emotion Scores (joy, fear, neutral, etc.)

### 5. Gradio Dashboard (`gradio-dashboard.py`)

> ***Objective***: Build an interactive book recommendation dashboard using Gradio. It leverages semantic search (powered by embeddings) and emotion-based filtering to recommend books based on user input, preferred categories, and emotional tones like joy, sadness, or suspense.

1. **Data Setup**
   - `books_with_emotions.csv` is loaded.
   - Adds large_thumbnail images — falling back to a placeholder if unavailable.
   - ✅ Nice touch: Ensures consistent image sizes with `&fife=w800`.
2. **Text Embeddings & Vector Database**
   - Uses `langchain` with Google Generative AI embeddings (`models/embedding-001`) for semantic search.
   - `Chroma` handles the vector database.
   - Supports persistent storage — skips re-embedding if the DB already exists (`chroma_db_books_vector_database`).
   - ✅ Efficiency boost: Faster reloads without retraining embeddings!
3. **Semantic Recommendations**
   - `retrieve_semantic_recommendations()` finds similar books.
   - Filters by category (optional).
   - Sorts by tone — prioritizes emotions (joy, fear, sadness, etc.).
   - ✅ Clever move: Sorting by emotional tone creates personalized results!
4. **Result Formatting**
   - `recommend_books()`:
     - Truncates long descriptions to 30 words.
     - Cleans up author names (e.g., "John, Jane, and Jack").
     - Outputs formatted gallery cards (cover + caption).
     - ✅ Polished UX: Shortened descriptions + clean author formatting keeps it readable.
5. **Gradio Dashboard**
   - Uses Gradio Blocks (with theme=gr.themes.Citrus()).
   - Includes:
     - Query Input
     - Category Dropdown
     - Tone Selector
     - Submit Button
     - Gallery Output (8 columns, 2 rows)
   - ✅ Looks professional: Modern, clean layout with citrus-themed Gradio blocks.
6. **Final Launch Setup**
   - `dashboard.launch()` on 0.0.0.0:7860 (ready for local or Docker deployment).
