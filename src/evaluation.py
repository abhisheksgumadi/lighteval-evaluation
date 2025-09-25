"""
Evaluation script using HuggingFace Lighteval framework
for evaluating fine-tuned LLMs on custom datasets.
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from datasets import Dataset
import argparse
from datetime import timedelta

# Lighteval imports
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import TransformersModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics import Metrics
from accelerate import Accelerator, InitProcessGroupKwargs

# Check if accelerate is available
try:
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
    is_accelerate_available = lambda: True
except ImportError:
    accelerator = None
    is_accelerate_available = lambda: False


def csv_prompt_fn(line, task_name: str = None):
    """
    Format the CSV data into a Doc object for evaluation.
    This function converts each row of your CSV into the format expected by Lighteval.
    """
    return Doc(
        task_name=task_name,
        query=line["input"],
        choices=[line["output"]],  # For generation tasks, we provide the expected output as choice
        gold_index=0,  # The correct answer is always at index 0
        target_for_fewshot=line["output"],  # Used for few-shot examples
        instruction="",  # Add instruction if your task needs it
    )


class CSVEvaluator:
    """
    Custom evaluator for CSV datasets using Lighteval framework.
    """
    
    def __init__(self, csv_path: str, input_column: str = "input", output_column: str = "output"):
        """
        Initialize the CSV evaluator.
        
        Args:
            csv_path: Path to the CSV file
            input_column: Name of the input column (default: "input")
            output_column: Name of the output column (default: "output")
        """
        self.csv_path = csv_path
        self.input_column = input_column
        self.output_column = output_column
        self.task_name = f"custom_test"
        
        # Load and validate the dataset
        self._load_and_validate_data()
        
        # Create temporary directory for the custom task
        self._create_custom_dataset()
    
    def _load_and_validate_data(self):
        """Load and validate the CSV data."""
        try:
            self.data = pd.read_csv(self.csv_path)
        except Exception as e:
            raise ValueError(f"Error loading CSV file {self.csv_path}: {e}")
        
        print(f"Loaded {len(self.data)} valid samples from {self.csv_path}")
    
    def _create_custom_dataset(self):
        """Create a custom task file for Lighteval."""
        # Convert pandas DataFrame to HuggingFace Dataset
        dataset = Dataset.from_pandas(self.data)
        
        # Create a proper dataset structure that can be loaded with load_dataset
        dataset_path = os.path.join("data", "custom_dataset")
        os.makedirs(dataset_path, exist_ok=True)
        
        # Save as JSON files that can be loaded with load_dataset
        train_file = os.path.join(dataset_path, "train.jsonl")
        with open(train_file, 'w') as f:
            for example in dataset:
                f.write(json.dumps(example) + '\n')
        
        print(f"Dataset saved to: {dataset_path}")
        
        # Create a new custom task file in the temp directory
        self.task_file_path = os.path.join("src/tasks", "custom_task.py")
        self.dataset_path = dataset_path
    
    def evaluate_model(
        self,
        model_name: str,
        output_dir: str = "./evaluation_results",
        batch_size: int = 1,
        max_samples: Optional[int] = None,
        push_to_hub: bool = False,
        hub_results_org: Optional[str] = None,
        override_chat_template: bool = False,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
    ) -> Dict[str, Any]:
        """
        Evaluate a model on the CSV dataset.
        
        Args:
            model_name: HuggingFace model name or path
            output_dir: Directory to save results
            batch_size: Batch size for evaluation
            max_samples: Maximum number of samples to evaluate
            push_to_hub: Whether to push results to Hub
            hub_results_org: Organization for Hub results
            override_chat_template: Whether to use chat template
            temperature: Generation temperature
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary containing evaluation results
        """
        
        # Set up evaluation tracker
        evaluation_tracker = EvaluationTracker(
            output_dir=output_dir,
            save_details=True,
            push_to_hub=push_to_hub,
            hub_results_org=hub_results_org,
        )
        
        # Set up pipeline parameters
        pipeline_params = PipelineParameters(
            launcher_type=ParallelismManager.ACCELERATE if is_accelerate_available() else ParallelismManager.SERIAL,
            custom_tasks_directory=self.task_file_path,
            max_samples=max_samples,
            dataset_loading_processes=1,
        )
        
        # Configure model using Transformers backend
        model_config = TransformersModelConfig(
            model_name=model_name,
            override_chat_template=override_chat_template,
            batch_size=batch_size,
            dtype="auto",
            generation_parameters={
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            }
        )
        
        # Create custom task string
        task_string = f"custom|custom_task|0|0"
        
        print(f"Starting evaluation of {model_name}")
        print(f"Task: {task_string}")
        print(f"Dataset size: {len(self.data)} samples")
        print(f"Max samples: {max_samples or 'all'}")
        print(f"Backend: transformers")
        
        # Create and run pipeline
        pipeline = Pipeline(
            tasks=task_string,
            pipeline_parameters=pipeline_params,
            evaluation_tracker=evaluation_tracker,
            model_config=model_config,
        )
        
        # Run evaluation
        pipeline.evaluate()
        
        # Save results
        pipeline.save_and_push_results()
        
        # Show results
        pipeline.show_results()
        


def evaluate_model_on_csv(
    model_name: str,
    csv_path: str,
    input_column: str = "input",
    output_column: str = "output",
    output_dir: str = "./evaluation_results",
    batch_size: int = 1,
    max_samples: Optional[int] = None,
    push_to_hub: bool = False,
    hub_results_org: Optional[str] = None,
    override_chat_template: bool = False,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """
    Convenience function to evaluate a model on CSV data.
    
    Args:
        model_name: HuggingFace model name or path
        csv_path: Path to CSV file with input/output pairs
        input_column: Name of input column
        output_column: Name of output column
        output_dir: Directory to save results
        batch_size: Batch size for evaluation
        max_samples: Maximum samples to evaluate
        push_to_hub: Whether to push results to Hub
        hub_results_org: Organization for Hub results
        override_chat_template: Whether to use chat template
        temperature: Generation temperature
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        Dictionary containing evaluation results
    """
    
    # Create evaluator
    evaluator = CSVEvaluator(
        csv_path=csv_path,
        input_column=input_column,
        output_column=output_column
    )
    
    # Run evaluation
    evaluator.evaluate_model(
        model_name=model_name,
        output_dir=output_dir,
        batch_size=batch_size,
        max_samples=max_samples,
        push_to_hub=push_to_hub,
        hub_results_org=hub_results_org,
        override_chat_template=override_chat_template,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Evaluate a HuggingFace model on custom CSV data using Lighteval")
    
    parser.add_argument("--model_name", type=str, default = "upstage/TinySolar-248m-4k", help="HuggingFace model name or path")
    parser.add_argument("--csv_path", type=str, default = "data/test.csv", help="Path to CSV file with input/output pairs")
    parser.add_argument("--input_column", type=str, default="input", help="Name of input column")
    parser.add_argument("--output_column", type=str, default="output", help="Name of output column")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--max_samples", default = 2, type=int, help="Maximum number of samples to evaluate")
    parser.add_argument("--push_to_hub", default = False, action="store_true", help="Push results to HuggingFace Hub")
    parser.add_argument("--hub_results_org", default = None, type=str, help="Organization name for pushing to Hub")
    parser.add_argument("--override_chat_template", default = False, action="store_true", help="Override chat template for model")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum tokens to generate")
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_model_on_csv(
        model_name=args.model_name,
        csv_path=args.csv_path,
        input_column=args.input_column,
        output_column=args.output_column,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        push_to_hub=args.push_to_hub,
        hub_results_org=args.hub_results_org,
        override_chat_template=args.override_chat_template,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )
    

if __name__ == "__main__":
    main()