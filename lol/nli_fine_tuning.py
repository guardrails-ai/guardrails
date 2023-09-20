def main_function(model_name, num_rows=10, num_epochs=3):
    import sys
    import os
    import subprocess

    # if on linux, install pysqlite3-binary and follow steps as mentioned here:
    # https://gist.github.com/defulmere/8b9695e415a44271061cc8e272f3c300
    if sys.platform in ("linux", "linux2"):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "pysqlite3-binary"]
        )
        __import__("pysqlite3")

        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
        BASE_DIR = os.getcwd()

        DATABASES = {
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": os.path.join(BASE_DIR, "db.sqlite3"),
            }
        }

    # Imports
    import pandas as pd
    from typing import List
    import nltk
    from nltk.tokenize import sent_tokenize
    from rich import print
    from tqdm import tqdm
    from datasets import Dataset
    import chromadb
    import wget
    import gzip
    import shutil
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )
    import numpy as np
    import evaluate

    # All helper functions
    # Set up dataset
    def get_dataset(path: str) -> pd.DataFrame:
        """Get the dataset.

        Does the following preprocessing:
        - Only keep the rows with complete_support or no_support
        - Only keep rows where the source_localized_evidence is a substring of the source_raw_text
        """

        # Read the jsonl file into pandas dataframe
        df = pd.read_json(path, lines=True)

        # Only keep the rows where the source_supports_statement is 'complete_support' or 'no_support'
        df = df.loc[
            df["source_supports_statement"].isin(["complete_support", "no_support"])
        ]

        # Only keep rows where the source_localized_evidence is a substring of the source_raw_text
        df["source_localized_evidence_in_source_raw_text"] = df.apply(
            lambda row: row["source_localized_evidence"] in row["source_raw_text"],
            axis=1,
        )
        df = df.loc[df["source_localized_evidence_in_source_raw_text"] == True]

        # Add a new column called 'supported' that is True if the source_supports_statement is 'complete_support' and False otherwise
        df["supported"] = df.apply(
            lambda row: 0
            if row["source_supports_statement"] == "complete_support"
            else 1,
            axis=1,
        )

        # Reset the index
        df = df.reset_index(drop=True)

        df = df[
            [
                "response",
                "source_text",
                "source_raw_text",
                "supported",
            ]
        ]
        return df

    # Helper functions to add docs to DB
    def get_chunks_from_text(
        text: str, chunk_strategy: str, chunk_size: int, chunk_overlap: int
    ) -> List[str]:
        """Get chunks of text from a string.

        Args:
            text: The text to chunk.
            chunk_strategy: The strategy to use for chunking. Currently only "sentence" and "paragraph" is supported.
            chunk_size: The size of each chunk. If the chunk_strategy is "sentences", this is the number of sentences per chunk.
            chunk_overlap: The number of characters to overlap between chunks.  If the chunk_strategy is "sentences", this is the number of sentences to overlap between chunks.
        """

        if chunk_strategy == "sentence":
            atomic_units = sent_tokenize(text)
        elif chunk_strategy == "paragraph":
            atomic_units = text.split("\n")
        else:
            raise ValueError(
                f"Invalid chunk strategy: {chunk_strategy}. Valid choices are 'sentence' and 'paragraph'."
            )

        if chunk_strategy == "sentence":
            chunks = []
            for i in range(0, len(atomic_units), chunk_size - chunk_overlap):
                chunk = " ".join(atomic_units[i : i + chunk_size])
                chunks.append(chunk)
        elif chunk_strategy == "paragraph":
            if chunk_size == 0:
                # No need to chunk paragraphs, just return a list of paragraphs
                chunks = atomic_units
            else:
                # Chunk paragraphs
                chunks = []
                for i in range(0, len(atomic_units), chunk_size - chunk_overlap):
                    chunk = " ".join(atomic_units[i : i + chunk_size])
                    chunks.append(chunk)
        return chunks

    def add_sources_to_vector_db(
        chroma_client, df, chunk_strategy, chunk_size, chunk_overlap, collection_name
    ):
        # Configure collection
        db_collection = chroma_client.get_or_create_collection(collection_name)

        if chunk_strategy == "sentence":
            # No need of formatted text for sentence chunks
            raw_texts = df["source_raw_text"]
        elif chunk_strategy == "paragraph":
            # Use formatted text for paragraph chunks to split based on "\n"
            raw_texts = df["source_text"]

        print(
            f"Adding chunks from {len(raw_texts)} documents to the vector database..."
        )
        for idx, text in tqdm(raw_texts.items()):
            chunks = get_chunks_from_text(
                text, chunk_strategy, chunk_size, chunk_overlap
            )
            db_collection.add(
                documents=chunks,
                metadatas=[
                    {"doc_id": idx, "chunk_id": chunk_id}
                    for chunk_id in range(len(chunks))
                ],
                ids=[f"{idx}-{chunk_id}" for chunk_id in range(len(chunks))],
            )
        return db_collection

    def get_most_relevant_chunk(row, collection, top_k=1):
        doc_id = row.name  # This is the index of the row.
        text = row["response"]

        # Get similar chunk from the vectorDB
        query_output = collection.query(
            query_texts=[text], n_results=top_k, where={"doc_id": doc_id}
        )
        relevant_chunk = query_output["documents"][0][0]

        return pd.Series([relevant_chunk], index=["relevant_chunk"])

    def create_final_dataset(
        path: str,
        chunk_strategy: str = "paragraph",
        chunk_size: int = 3,
        chunk_overlap: int = 0,
    ):
        # Get the dataset
        df = get_dataset(path)
        df = df.iloc[:num_rows]

        # Create a new ChromaDB client
        chroma_client = chromadb.Client()

        # Chunk up source text and add to vector db
        # Add docs to vector DB collection
        collection_name = f"{chunk_strategy}_{chunk_size}_{chunk_overlap}"
        collection = add_sources_to_vector_db(
            chroma_client,
            df,
            chunk_strategy,
            chunk_size,
            chunk_overlap,
            collection_name,
        )

        df = df[
            [
                "response",
                "supported",
            ]
        ]

        # Get the most relevant chunk for each response and add that to new column in df
        df[["relevant_chunks"]] = df.apply(
            get_most_relevant_chunk, axis=1, args=(collection, 1)
        )

        # Rename columns
        df.rename(
            columns={
                "relevant_chunks": "premise",
                "response": "hypothesis",
                "supported": "label",
            },
            inplace=True,
        )

        # Create datasets.DatasetDict with train-dev split
        dataset = Dataset.from_pandas(df)
        dataset = dataset.train_test_split(test_size=0.2)

        # Delete collection
        chroma_client.delete_collection(collection_name)
        # Return that
        return dataset

    #### Main code ####
    nltk.download("punkt")

    # Download and unzip the dataset and then delete later
    dataset_name = "verifiability_judgments_train.jsonl"
    url = "https://github.com/nelson-liu/evaluating-verifiability-in-generative-search-engines/raw/main/verifiability_judgments/verifiability_judgments_train.jsonl.gz"
    wget.download(url, dataset_name + ".gz")
    with gzip.open(dataset_name + ".gz", "rb") as f_in:
        with open(dataset_name, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    print("Dataset downloaded and unzipped successfully!")

    dataset = create_final_dataset(dataset_name)
    dataset_train, dataset_dev = dataset["train"], dataset["test"]
    print("Dataset created successfully!")

    # Delete the downloaded dataset
    os.remove(dataset_name + ".gz")
    os.remove(dataset_name)

    # Define NLI model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def tokenize_function(example):
        # Premise -> hypothesis pair
        return tokenizer(
            example["premise"],
            example["hypothesis"],
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )

    dataset_train_tok = dataset_train.map(tokenize_function).shuffle(seed=42)
    dataset_dev_tok = dataset_dev.map(tokenize_function).shuffle(seed=42)
    print("Tokenized successfully!")

    # Define the model
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    training_args = TrainingArguments(
        output_dir=f"{model_name}-output",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        num_train_epochs=num_epochs,
        save_total_limit=1,
        load_best_model_at_end=True,
    )

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train_tok,
        eval_dataset=dataset_dev_tok,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("Training completed successfully!")

    # Save the model
    path = f"{model_name}-trained"
    trainer.save_model(path)
    print("Model saved successfully!")

    # Delete the output directory
    shutil.rmtree(f"{model_name}-output")
    print("Output directory deleted successfully!")
    return path
