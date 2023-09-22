import sys
import os
import subprocess

# if on linux, install pysqlite3-binary and follow steps as mentioned here:
# https://gist.github.com/defulmere/8b9695e415a44271061cc8e272f3c300
if sys.platform in ("linux", "linux2"):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pysqlite3-binary"])
    __import__("pysqlite3")

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    BASE_DIR = os.getcwd()

    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": os.path.join(BASE_DIR, "db.sqlite3"),
        }
    }
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"])
##############################################
# Imports
##############################################
from typing import List, Tuple
import chromadb
import pandas as pd
from chromadb.utils import embedding_functions
import nltk
from nltk.tokenize import sent_tokenize
from rich import print
from tqdm import tqdm
import itertools
import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import wget
import shutil
import gzip

nltk.download("punkt")
##############################################
# Global variables
##############################################

# Experiment config
CHUNK_STRATEGIES = ["sentence", "paragraph"]
CHUNK_SIZES = [1, 3]  # Number of sentences/paragraphs per chunk
CHUNK_OVERLAP = 0
EMBEDDING_MODELS = ["L6"]
DISTANCE_METRIC = "cosine"
TOP_K_VALUES = [1, 3, 5]  # Number of similar chunks to retrieve (and then use as premises for the NLI model)
NLI_MODELS = [
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c",
]

EMBEDDING_MODEL_TO_NAME = {
    "L6": "all-MiniLM-L6-v2",
}

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

##############################################
# All methods
##############################################


# Set up dataset
def get_dataset(path: str) -> pd.DataFrame:
    """Get the dataset for the provenance testing.

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
        lambda row: row["source_localized_evidence"] in row["source_raw_text"], axis=1
    )
    df = df.loc[df["source_localized_evidence_in_source_raw_text"] == True]

    # Add a new column called 'supported' that is True if the source_supports_statement is 'complete_support' and False otherwise
    df["supported"] = df.apply(
        lambda row: row["source_supports_statement"] == "complete_support", axis=1
    )

    # Reset the index
    df = df.reset_index(drop=True)

    df = df[
        [
            "query",
            "response",
            "statement",
            "source_title",
            "source_content_title",
            "source_text",
            "source_raw_text",
            "source_localized_evidence",
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
    elif chunk_strategy == "document":
        # No need to split
        atomic_units = [text]
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
    elif chunk_strategy == "document":
        chunks = atomic_units
    return chunks


def add_sources_to_vector_db(df, collection, chunk_strategy, chunk_size):
    if collection.count() > 0:
        print(
            f"Collection {collection.name} already has {collection.count()} documents. Skipping..."
        )
        return
    else:
        if chunk_strategy in ("sentence", "document"):
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
                text, chunk_strategy, chunk_size, CHUNK_OVERLAP
            )
            collection.add(
                documents=chunks,
                metadatas=[
                    {"doc_id": idx, "chunk_id": chunk_id}
                    for chunk_id in range(len(chunks))
                ],
                ids=[f"{idx}-{chunk_id}" for chunk_id in range(len(chunks))],
            )


def get_nli_response(tokenizer, model, premise: str, hypothesis: str, nli_model_name: str) -> str:
    """Get the response from the HuggingFace NLI model.

    Args:
        premise: The premise to use for the NLI model.
        hypothesis: The hypothesis to use for the NLI model.
        nli_model_name: The name of the NLI model to use.

    Returns:
        str: True if the NLI model predicts entailment, False otherwise.
    """
    if nli_model_name == "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli":
        tokenized_input_seq_pair = tokenizer.encode_plus(
            premise,
            hypothesis,
            # max_length=max_length,
            return_token_type_ids=True,
            truncation=True,
        )

        input_ids = (
            torch.Tensor(tokenized_input_seq_pair["input_ids"]).long().unsqueeze(0)
        )
        # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
        token_type_ids = (
            torch.Tensor(tokenized_input_seq_pair["token_type_ids"]).long().unsqueeze(0)
        )
        attention_mask = (
            torch.Tensor(tokenized_input_seq_pair["attention_mask"]).long().unsqueeze(0)
        )

        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=None,
        )

        id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}

        predicted_probabilities = torch.softmax(outputs[0], dim=1)[
            0
        ].tolist()  # batch_size only one

        max_label = id2label[
            predicted_probabilities.index(max(predicted_probabilities))
        ]
        confidence = round(max(predicted_probabilities), 3)
        return (
            (
                True,
                confidence,
            )  # Here, confidence value is the probability of entailment
            if max_label == "entailment"
            else (
                False,
                round(
                    predicted_probabilities[1] + predicted_probabilities[2], 3
                ),  # Here, confidence value is the probability of contradiction + neutral
            )
        )
    elif nli_model_name == "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c":
        input = tokenizer(
            premise, hypothesis, padding="max_length", truncation=True, return_tensors="pt"
        )
        output = model(
            input["input_ids"]
        )  # device = "cuda:0" or "cpu"
        prediction = torch.softmax(output["logits"][0], -1).tolist()
        label_names = ["entailment", "not_entailment"]
        prediction = {name: pred for pred, name in zip(prediction, label_names)}
        max_label = max(prediction, key=prediction.get)
        confidence = round(prediction[max_label], 3)
        return (True, confidence) if max_label == "entailment" else (False, confidence)


def evaluate_with_nli_model(
    row: pd.Series, top_k: int, nli_model_name: str, collection, tokenizer, model
) -> pd.Series:
    """For each response, retrieves the similar chunks of text from the vectorDB,
    creates the prompt for the LLM that includes the response and the similar chunks,
    and makes OpenAI API calls using that prompt to output a boolean value
    indicating whether the response is relevant to the similar chunks.

    Returns:
        pd.Series: A series of the retrieved chunks, the prompt, the self-evaluation
    """
    doc_id = row.name  # This is the index of the row.
    text = row["response"]

    # Get similar chunks from the vectorDB
    query_output = collection.query(
        query_texts=[text], n_results=top_k, where={"doc_id": doc_id}
    )
    chunks = query_output["documents"][0]

    # For each retrieved chunk, get NLI response with
    # premise: chunk, hypothesis: text (LLM response)
    # Return the best self-evaluation based on the confidence value
    max_confidence = 0
    best_self_eval = None
    best_chunk = None
    best_chunk_id = 0
    for i, chunk in enumerate(chunks):
        self_eval, confidence = get_nli_response(
            tokenizer, model, premise=chunk, hypothesis=text, nli_model_name=nli_model_name
        )
        if confidence > max_confidence:
            max_confidence = confidence
            best_self_eval = self_eval
            best_chunk = chunk
            best_chunk_id = i

    return pd.Series(
        [best_chunk, best_chunk_id, best_self_eval, max_confidence],
        index=["chunk", "chunk_id", "self_eval", "confidence"],
    )


##############################################
# Main method
##############################################

if __name__ == "__main__":
    # Get dataset
    # Download and unzip the dataset and then delete later
    dataset_name = "verifiability_judgments_dev.jsonl"
    url = "https://github.com/nelson-liu/evaluating-verifiability-in-generative-search-engines/raw/main/verifiability_judgments/verifiability_judgments_dev.jsonl.gz"
    wget.download(url, dataset_name + ".gz")
    with gzip.open(dataset_name + ".gz", "rb") as f_in:
        with open(dataset_name, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    print("Dataset downloaded and unzipped successfully!")

    df = get_dataset(dataset_name)
    print("Dataset loaded.")

    # Delete the downloaded dataset
    os.remove(dataset_name + ".gz")
    os.remove(dataset_name)

    # Create a new ChromaDB client
    chroma_client = chromadb.Client()
    print("Created ChromaDB client.")

    # Create a grid of all possible combinations
    combinations = list(
        itertools.product(
            CHUNK_STRATEGIES, CHUNK_SIZES, EMBEDDING_MODELS, TOP_K_VALUES, NLI_MODELS
        )
    )
    print(f"Total number of combinations to test: {len(combinations)}")

    for combination in combinations:
        try:
            config = {
                "chunk_strategy": combination[0],
                "chunk_size": combination[1],
                "embedding_model": combination[2],
                "top_k": combination[3],
                "nli_model_name": combination[4],
            }

            print(config)

            embed_func = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL_TO_NAME[config["embedding_model"]]
            )

            # Configure collection
            config_str = f"{config['chunk_strategy']}_{config['chunk_size']}_{config['embedding_model']}"
            db_collection = chroma_client.get_or_create_collection(
                config_str,
                metadata=config,
                embedding_function=embed_func,
            )

            # Add docs to vector DB collection
            add_sources_to_vector_db(
                df, db_collection, config["chunk_strategy"], config["chunk_size"]
            )
            print(f"Added sources to {db_collection.name}.")

            print(f"Evaluating responses for {db_collection.name}...")
            tokenizer = AutoTokenizer.from_pretrained(config["nli_model_name"])
            model = AutoModelForSequenceClassification.from_pretrained(f"{config['nli_model_name']}-trained-{config['chunk_strategy']}-{config['chunk_size']}")
            df[
                [
                    f"chunk_{db_collection.name}",
                    f"chunk_id_{db_collection.name}",
                    f"self_evaluation_{db_collection.name}",
                    f"confidence_{db_collection.name}",
                ]
            ] = df.apply(
                evaluate_with_nli_model,
                axis=1,
                args=(config["top_k"], config["nli_model_name"], db_collection, tokenizer, model),
            )
            print(f"Evaluated responses for {db_collection.name}.")

            # Get the percentage of times when the 0 chunk_id was selected
            chunk0_percentage = round(
                df[f"chunk_id_{db_collection.name}"].value_counts()[0]
                / df.shape[0]
                * 100,
                3,
            )

            # Get min and average confidence value for the df
            min_confidence = round(df[f"confidence_{db_collection.name}"].min(), 3)
            avg_confidence = round(df[f"confidence_{db_collection.name}"].mean(), 3)

            # Create positive and negative dataframes for analysis
            df_positive = df.loc[df["supported"] == True]
            df_negative = df.loc[df["supported"] == False]

            # FPR, TPR based on 'supported' and 'self-evaluations'
            try:
                tp = df_positive[
                    f"self_evaluation_{db_collection.name}"
                ].value_counts()[True]
            except KeyError:
                tp = 0

            try:
                tn = df_negative[
                    f"self_evaluation_{db_collection.name}"
                ].value_counts()[False]
            except KeyError:
                tn = 0

            try:
                fp = df_negative[
                    f"self_evaluation_{db_collection.name}"
                ].value_counts()[True]
            except KeyError:
                fp = 0

            try:
                fn = df_positive[
                    f"self_evaluation_{db_collection.name}"
                ].value_counts()[False]
            except KeyError:
                fn = 0

            # True positive rate
            tpr = round(tp / len(df_positive), 2)
            print(f"True positive rate: {(tpr * 100):.2f}%")

            # False positive rate
            fpr = round(fp / len(df_negative), 2)
            print(f"False positive rate: {(fpr * 100):.2f}%")

            # True negative rate
            tnr = round(tn / len(df_negative), 2)
            print(f"True negative rate: {(tnr * 100):.2f}%")

            # False negative rate
            fnr = round(fn / len(df_positive), 2)
            print(f"False negative rate: {(fnr * 100):.2f}%")

            try:
                precision = round(tp / (tp + fp), 2)
            except ZeroDivisionError:
                precision = 0.0

            try:
                recall = round(tp / (tp + fn), 2)
            except ZeroDivisionError:
                recall = 0.0

            try:
                f1 = round(2 * (precision * recall) / (precision + recall), 2)
            except ZeroDivisionError:
                f1 = 0.0

            try:
                accuracy = round((tp + tn) / (tp + tn + fp + fn), 2)
            except ZeroDivisionError:
                accuracy = 0.0

            print(f"Precision: {precision:.2f}")
            print(f"Recall: {recall:.2f}")
            print(f"F1: {f1:.2f}")
            print(f"Accuracy: {accuracy * 100:.2f}%")

            # Add a new row to csv file
            with open(
                "outputs/v2-outputs.csv",
                "a",
            ) as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        config["nli_model_name"],
                        config["chunk_strategy"],
                        CHUNK_OVERLAP,
                        DISTANCE_METRIC,
                        config["chunk_size"],
                        EMBEDDING_MODEL_TO_NAME[config["embedding_model"]],
                        config["top_k"],
                        tpr,
                        fpr,
                        tnr,
                        fnr,
                        precision,
                        recall,
                        f1,
                        accuracy,
                        chunk0_percentage,
                        min_confidence,
                        avg_confidence,
                    ]
                )

            print(f"Results written to csv file for combination: {config}")
            print(
                "##########################################################################################"
            )
        except Exception as e:
            print(f"Error for combination: {config}")
            print(e)
            print(
                "##########################################################################################"
            )
            continue

