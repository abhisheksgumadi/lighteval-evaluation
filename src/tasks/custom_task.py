from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics import Metrics

def csv_prompt_fn(line, task_name: str = None):
    """Format CSV data into Doc object."""
    return Doc(
        task_name=task_name,
        query=line["input"],
        choices=[line["output"]],
        gold_index=0,
        instruction="",
    )

# Task configuration
custom_task = LightevalTaskConfig(
    name="custom_task",
    prompt_function=csv_prompt_fn,
    hf_repo="data/custom_dataset",
    hf_subset="",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    suite=["custom"],
    generation_size=512,
    metrics=[Metrics.exact_match, Metrics.rouge1, Metrics.rouge2],
    stop_sequence=[],
    version=0,
)

# Register the task
TASKS_TABLE = [custom_task]