"""Semantic book recommender dashboard (Gradio).


Combines vector search with category and emotional-tone filters.
Run: python new_gradio-dashboard.py
"""


import os


import gradio as gr
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()


books = pd.read_csv("datasets/books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)


with open("datasets/tagged_description.txt", encoding="utf-8") as f:
    documents = [Document(page_content=line.strip()) for line in f if line.strip()]


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_books = Chroma.from_documents(documents, embedding=embedding_model)


TONE_COLUMNS = {
    "Happy": "joy",
    "Surprising": "surprise",
    "Angry": "anger",
    "Suspenseful": "fear",
    "Sad": "sadness",
}




def retrieve_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50,
    final_top_k: int = 16,
) -> pd.DataFrame:
    """Semantic search with optional category and emotion filtering."""
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)


    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)


    if tone in TONE_COLUMNS:
        book_recs = book_recs.sort_values(by=TONE_COLUMNS[tone], ascending=False)


    return book_recs




def recommend_books(query: str, category: str, tone: str) -> list[tuple[str, str]]:
    """Return (thumbnail_url, caption) pairs for the Gradio Gallery."""
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []


    for _, row in recommendations.iterrows():
        truncated_description = " ".join(row["description"].split()[:30]) + "..."


        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]


        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))


    return results




categories = ["All"] + sorted(
    cat for cat in books["simple_categories"].unique() if pd.notna(cat) and cat != ""
)
tones = ["All"] + list(TONE_COLUMNS)


with gr.Blocks() as dashboard:
    gr.Markdown("# Semantic Book Recommender")
    gr.Markdown(
        "Describe the kind of book you want. Optionally filter by **category** "
        "(Fiction/Nonfiction) or **emotional tone** (Happy, Sad, etc.)."
    )


    with gr.Row():
        user_query = gr.Textbox(
            label="Book description",
            placeholder="e.g., A story about forgiveness",
            scale=2,
        )
        category_dropdown = gr.Dropdown(choices=categories, label="Category", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Emotional tone", value="All")
        submit_button = gr.Button("Find recommendations", variant="primary")


    gr.Markdown("## Recommendations")
    output = gr.Gallery(
        label="Recommended books",
        columns=4,
        rows=2,
        height="auto",
        object_fit="contain",
    )


    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output,
    )


host = os.getenv("HOST")
port = int(os.getenv("PORT", 7860))


if __name__ == "__main__":
    launch_kwargs = {"theme": gr.themes.Glass()}
    if host:
        launch_kwargs["server_name"] = host
        launch_kwargs["server_port"] = port
    else:
        launch_kwargs["server_port"] = port
    dashboard.launch(**launch_kwargs)



