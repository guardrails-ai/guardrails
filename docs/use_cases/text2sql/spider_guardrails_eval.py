import argparse
from calendar import c
from collections import defaultdict
import os
from pathlib import Path
import json
from guardrails.embedding import ManifestEmbedding
from guardrails.applications.text2sql import Text2Sql
from manifest import Manifest
from pathlib import Path
from tqdm.auto import tqdm
from typing import Any, Callable, List, Dict, Optional
import sys
from rich.console import Console

console = Console(soft_wrap=True)

SPIDER_EVAL_PATH = os.environ.get("SPIDER_EVAL_DIR")
if SPIDER_EVAL_PATH is None:
    raise ValueError(
        "Please set SPIDER_EVAL_DIR to the path of the spider evaluation code"
    )
sys.path.append(SPIDER_EVAL_PATH)
from spider_metrics.spider import evaluation as spider_evaluation  # type: ignore
from spider_metrics.test_suite_sql_eval import (  # type: ignore
    evaluation as test_suite_evaluation,
)

os.environ["TIKTOKEN_CACHE_DIR"] = str(Path.home() / ".cache/tiktoken")

REASK_PROMPT = """
You are a data scientist whose job is to write SQL queries.

Here's schema about the database that you can use to generate the SQL query.
Try to avoid using joins if the data can be retrieved from the same table.

{{db_info}}

I will give you a list of examples.

{{examples}}
I want to create a query for the following instruction:

Instruction: {{nl_instruction}}

For this instruction, I was given the following SQL, which has some incorrect values.

Original SQL: {previous_response}
Error: {error_messages}

Output the corrected SQL below. Just give the SQL. Start with a SELECT.
SQL:"""


def sql_example_formatter(
    input: str,
    output: str,
) -> str:
    example = f"Instruction: {input}\nSQL: {output}\n"
    return example


def compute_exact_match_metric(
    predictions: List, references: List, gold_dbs: List, kmaps: Dict, db_dir: str
) -> tuple[Any, List[int | None]]:
    """Compute exact match metric."""
    evaluator = spider_evaluation.Evaluator(db_dir, kmaps, "match")
    by_row_metrics: list[int | None] = []
    for prediction, reference, gold_db in tqdm(
        zip(predictions, references, gold_dbs), total=len(predictions)
    ):
        turn_idx = 0
        # skip final utterance-query pairs
        if turn_idx < 0:
            continue
        try:
            em_metrics = evaluator.evaluate_one(gold_db, reference, prediction)
            by_row_metrics.append(int(em_metrics["exact"]))
        except Exception as e:
            print(e)
            by_row_metrics.append(None)
            pass
    evaluator.finalize()
    return evaluator.scores["all"]["exact"], by_row_metrics


def compute_test_suite_metric(
    predictions: List, references: List, gold_dbs: List, kmaps: Dict, db_dir: str
) -> tuple[Any, List[int | None]]:
    """Compute test suite execution metric."""
    evaluator = test_suite_evaluation.Evaluator(
        db_dir=db_dir,
        kmaps=kmaps,
        etype="exec",
        plug_value=False,
        keep_distinct=False,
        progress_bar_for_each_datapoint=False,
    )
    # Only used for Sparc/CoSQL
    turn_scores: dict[str, list] = {"exec": [], "exact": []}
    by_row_metrics: list[int | None] = []
    for prediction, reference, gold_db in tqdm(
        zip(predictions, references, gold_dbs), total=len(predictions)
    ):
        turn_idx = 0
        # skip final utterance-query pairs
        if turn_idx < 0:
            continue
        try:
            ex_metrics = evaluator.evaluate_one(
                gold_db,
                reference,
                prediction,
                turn_scores,
                idx=turn_idx,
            )
            by_row_metrics.append(int(ex_metrics["exec"]))
        except Exception as e:
            print(e)
            by_row_metrics.append(None)
            pass
    evaluator.finalize()
    return evaluator.scores["all"]["exec"], by_row_metrics


def compute_metrics(
    pred_sqls: list[str],
    max_run: int = -1,
    spider_dir: str = "spider_data",
) -> dict[str, str]:
    """Compute all metrics for data slice."""
    gold_path = Path(spider_dir) / "spider" / "dev_gold.sql"
    gold_input_path = Path(spider_dir) / "spider" / "dev.json"
    tables_path = Path(spider_dir) / "spider" / "tables.json"
    database_dir = str(Path(spider_dir) / "spider" / "database")

    kmaps = test_suite_evaluation.build_foreign_key_map_from_json(str(tables_path))
    gold_sqls, gold_dbs = zip(
        *[l.strip().split("\t") for l in gold_path.open("r").readlines()]
    )
    gold_sql_dict = json.load(gold_input_path.open("r"))

    # Subselect
    if max_run > 0:
        print(f"Subselecting {max_run} examples")
        gold_sqls = gold_sqls[:max_run]
        gold_dbs = gold_dbs[:max_run]
        gold_sql_dict = gold_sql_dict[:max_run]
        pred_sqls = pred_sqls[:max_run]
    # Data validation
    assert len(gold_sqls) == len(
        pred_sqls
    ), "Sample size doesn't match between pred and gold file"
    assert len(gold_sqls) == len(
        gold_sql_dict
    ), "Sample size doesn't match between gold file and gold dict"
    all_metrics: dict[str, Any] = {}

    # Execution Accuracy
    metrics, by_row_metrics_exec = compute_test_suite_metric(
        pred_sqls, gold_sqls, gold_dbs, kmaps, database_dir
    )
    all_metrics["exec"] = metrics

    # Exact Match Accuracy
    metrics, by_row_metrics_exact = compute_exact_match_metric(
        pred_sqls, gold_sqls, gold_dbs, kmaps, database_dir
    )
    all_metrics["exact"] = metrics

    # Merge all results into a single dict
    for i in range(len(gold_sql_dict)):
        gold_sql_dict[i]["pred"] = pred_sqls[i]
        gold_sql_dict[i]["exec"] = by_row_metrics_exec[i]
        gold_sql_dict[i]["exact"] = by_row_metrics_exact[i]
        gold_sql_dict[i]["gold"] = gold_sqls[i]
    return all_metrics, gold_sql_dict


def run_eval(pred_sqls, output_dir, max_run, spider_dir):
    """Run evaluation."""
    all_metrics, gold_sql_dict = compute_metrics(pred_sqls, max_run, spider_dir)
    # Write results to file
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing results to {output_dir}")
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f)
    with open(output_dir / "dump.json", "w") as f:
        json.dump(gold_sql_dict, f)
    return all_metrics


def main():
    args = parse_args()

    print(json.dumps(vars(args), indent=2))

    root = Path(args.spider_dir)
    train_data = json.load(open(root / "spider/train_spider.json"))
    dev_data = json.load(open(root / "spider/dev.json"))
    tables = json.load(open(root / "spider/tables.json"))
    num_run = min(args.num_run, len(dev_data)) if args.num_run > 0 else len(dev_data)
    num_demonstrations = args.num_demonstrations
    # print("**Example**")
    # print(json.dumps(dev_data[0], indent=2))
    print(f"{len(dev_data)} dev examples")

    embedding_manifest = ManifestEmbedding(
        client_name="huggingfaceembedding",
        client_connection="http://127.0.0.1:5000",
        cache_name="sqlite",
        cache_connection="spider_guard_cache_emb.db",
    )

    manifest = Manifest(
        client_name="openaichat",
        engine=args.gpt_model,
        cache_name="sqlite",
        cache_connection="spider_guard_cache.db",
    )

    if not args.indb:
        examples = [
            {"question": ex["question"], "query": ex["query"]} for ex in train_data
        ]
        examples_per_db = {"default": examples}
    else:
        examples_per_db = defaultdict(list)
        for ex in dev_data:
            dct = {"question": ex["question"], "query": ex["query"]}
            examples_per_db[ex["db_id"]].append(dct)

    # Iterate over all unique database ids
    database_ids = set([ex["db_id"] for ex in dev_data])
    all_apps = {}
    # Use the same store for all databases
    store = None
    if args.noreask:
        rail_file = "text2sql2_noreask.rail"
    else:
        rail_file = "text2sql2.rail"
    rail_spec = os.path.join(os.path.dirname(__file__), rail_file)
    for i, db_id in enumerate(database_ids):
        conn_str = f"sqlite:///{root}/spider/database/{db_id}/{db_id}.sqlite"
        if i == 0 or args.indb:
            app = Text2Sql(
                conn_str=conn_str,
                examples=examples_per_db[db_id]
                if args.indb
                else examples_per_db["default"],
                rail_spec=rail_spec,
                embedding=embedding_manifest,
                example_formatter=sql_example_formatter,
                reask_prompt=REASK_PROMPT,
                llm_api=manifest,
                num_relevant_examples=num_demonstrations,
            )
            store = app.store
        else:
            app = Text2Sql(
                conn_str=conn_str,
                examples=None,
                rail_spec=rail_spec,
                embedding=embedding_manifest,
                example_formatter=sql_example_formatter,
                reask_prompt=REASK_PROMPT,
                llm_api=manifest,
                num_relevant_examples=num_demonstrations,
            )
            assert store is not None
            app.store = store
        all_apps[db_id] = app

    pred_sqls = []
    gold_sqls = []
    for dev_ex in tqdm(dev_data[:num_run], desc="Predicting"):
        database_id = dev_ex["db_id"]
        if database_id not in all_apps:
            continue
        app = all_apps[database_id]
        question = dev_ex["question"]
        gold_query = dev_ex["query"]
        pred_query = app(question, gold_query)
        if not pred_query:
            print("BAD QUESTION", question)
            if app.guard.state.most_recent_call:
                print(console.print(app.guard.state.most_recent_call.tree))
        pred_sqls.append(pred_query or "")
        gold_sqls.append(gold_query)

    print("Running eval")
    output_dir = (
        Path(args.output_dir)
        / f"{args.gpt_model}_{num_run}n_{num_demonstrations}d_{int(args.indb)}indb_{int(args.noreask)}noreask"
    )
    metrics = run_eval(
        pred_sqls=pred_sqls,
        output_dir=output_dir,
        max_run=num_run,
        spider_dir=str(root),
    )
    print(json.dumps(metrics, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_run", type=int, default=200)
    parser.add_argument("--num_demonstrations", type=int, default=0)
    parser.add_argument("--indb", action="store_true")
    parser.add_argument("--noreask", action="store_true")
    parser.add_argument(
        "--spider_dir",
        type=str,
        required=True,
    )
    parser.add_argument("--output_dir", type=str, default="spider_eval_output")
    parser.add_argument(
        "--gpt_model",
        type=str,
        default="gpt-3.5-turbo",
        choices=["gpt-3.5-turbo", "gpt-4"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
