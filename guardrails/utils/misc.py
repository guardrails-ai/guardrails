import os
import random
from typing import List

from lxml.builder import E
from rich.pretty import pretty_repr

from guardrails import datatypes as dt
from guardrails.utils.logs_utils import GuardHistory
from guardrails.utils.reask_utils import gather_reasks


def generate_test_artifacts(
    rail_spec: str, guard_history: GuardHistory, on_fail_type: str, artifact_dir: str
) -> None:
    """Generate artifacts for testing.

    Artifacts include: rail_spec, compiled_prompt, llm_output, validated_response.
    The artifacts are saved by on_fail_type. Check out
    tests/integration_tests/test_assets/entity_extraction/ for examples.

    This function is only intended to be used to create artifacts for integration tests
    once the GuardHistory object has been manually checked to be correct.

    Args:
        rail_spec: This should be a string representation of the rail.
        guard_history: The guard history object.
        on_fail_type: The type of action to take when a validator fails.
        artifact_dir: The artifact dir where the artifacts will be saved.
    """

    # Create the artifact dir if it doesn't exist.
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    # Save the rail spec.
    with open(os.path.join(artifact_dir, f"{on_fail_type}.rail"), "w") as f:
        f.write(rail_spec)

    for i, logs in enumerate(guard_history.history):
        if i == 0:
            ext = ""
        elif i == 1:
            if len(guard_history.history) == 2:
                ext = "_reask"
            else:
                ext = "_reask_1"
        else:
            ext = f"_reask_{i}"

        # Save the compiled prompt.
        compiled_prompt = logs.prompt
        with open(
            os.path.join(artifact_dir, f"compiled_prompt_{on_fail_type}{ext}.txt"), "w"
        ) as f:
            f.write(compiled_prompt)

        # Save the llm output.
        llm_output = logs.output
        with open(
            os.path.join(artifact_dir, f"llm_output_{on_fail_type}{ext}.txt"), "w"
        ) as f:
            f.write(llm_output)

        # Save the validated response.
        validated_output = logs.validated_output
        with open(
            os.path.join(artifact_dir, f"validated_response_{on_fail_type}{ext}.py"),
            "w",
        ) as f:
            f.write("# flake8: noqa: E501\n")

            reasks = gather_reasks(validated_output)
            if len(reasks):
                f.write("from guardrails.utils.reask_utils import ReAsk\n")

            validated_output_repr = pretty_repr(validated_output, max_string=None)
            f.write(f"\nVALIDATED_OUTPUT = {validated_output_repr}")


def generate_random_schemas(n: int, depth: int = 4, width: int = 10) -> List[str]:
    """Generate random schemas that represent a valid schema.

    Args:
        n: The number of schemas to generate.
        depth: The depth of nesting
    """

    def random_scalar_datatype():
        selected_datatype = random.choice(
            [
                dt.String,
                dt.Integer,
                dt.Float,
                dt.Boolean,
                dt.Date,
                dt.Time,
            ]
        )
        return selected_datatype(None, None, None).rail_alias

    def generate_schema(curr_depth):
        if curr_depth < depth:
            # Type of current node is choice between "object", "list", "scalar"
            node_type = random.choice(["object", "list", "scalar"])

            if node_type == "object":
                # If "object", then generate random number of children
                num_children = random.randint(1, width)
                children = []
                for _ in range(num_children):
                    children.append(generate_schema(curr_depth + 1))
                return E.object(
                    *children, name=f"random_object_{random.randint(0, 1000)}"
                )
            elif node_type == "list":
                # If "list", then generate a single child
                return E.list(
                    generate_schema(curr_depth + 1),
                    name=f"random_list_{random.randint(0, 1000)}",
                )
            else:
                # If "scalar", then return a random primitive type
                datatype = random_scalar_datatype()
                return E(datatype, name=f"random_{datatype}_{random.randint(0, 1000)}")

        else:
            datatype = random_scalar_datatype()
            return E(datatype, name=f"random_{datatype}_{random.randint(0, 1000)}")

    schemas = []
    for _ in range(n):
        root = E("output")
        children = []
        num_children = random.randint(1, width)
        for _ in range(num_children):
            children.append(generate_schema(curr_depth=1))
        root.extend(children)

        schemas.append(root)
    return schemas
