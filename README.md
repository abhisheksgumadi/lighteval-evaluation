# Tutorial on using HugingFace Lighteval Evaluation Framework on Custom Data to Evaluate LLMs

Lighteval A comprehensive evaluation framework for testing Large Language Models (LLMs) on custom datasets. This project provides an easy-to-use turorial for evaluating fine-tuned or pre-trained models on your own CSV datasets with metric computation using the Lighteval framework. Think of this as a supporting turorial in addition to the one on the excellent Hugging Face documentation. This also accompanies the blog at https://medium.com/@abhisheksgumadi/evaluating-large-language-models-on-custom-data-using-hugging-face-lighteval-framework-132609ce8bf9

## 📋 Requirements

- Python 3.13+
- Poetry (for dependency management)

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd lighteval-evaluation
   ```

2. **Install dependencies using Poetry:**
   ```bash
   poetry install
   ```

3. **Activate the virtual environment:**
   ```bash
   poetry shell
   ```

## 📊 Dataset Format

Your CSV dataset should have the following structure:

```csv
input,output
"The weather is beautiful today and the sun is shining brightly.","I love sunny days when I can go for a walk in the park."
"Machine learning algorithms are becoming more sophisticated every year.","AI technology continues to advance rapidly with new breakthroughs."
```

**Required columns:**
- `input`: The input text/prompt for the model
- `output`: The expected output/response from the model

## 🏃‍♂️ Quick Start

### Basic Usage

Evaluate a model on the default test dataset:

```bash
python src/evaluation.py
```

This will:
- Use the default model: `upstage/TinySolar-248m-4k` for illustration purposes.
- Evaluate on `data/test.csv`
- Process 2 samples (for quick testing)
- Save results to `./evaluation_results/`

### Custom Model and Dataset

```bash
python src/evaluation.py \
    --model_name "microsoft/DialoGPT-medium" \
    --csv_path "path/to/your/dataset.csv" \
    --max_samples 100 \
    --batch_size 4
```

### Full Evaluation

```bash
python src/evaluation.py \
    --model_name "meta-llama/Llama-2-7b-chat-hf" \
    --csv_path "data/your_dataset.csv" \
    --max_samples 1000 \
    --batch_size 8 \
    --temperature 0.1 \
    --max_new_tokens 256
```

## ⚙️ Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name` | `upstage/TinySolar-248m-4k` | Hugging Face model name or local path |
| `--csv_path` | `data/test.csv` | Path to your CSV dataset |
| `--input_column` | `input` | Name of the input column |
| `--output_column` | `output` | Name of the output column |
| `--output_dir` | `./evaluation_results` | Directory to save results |
| `--batch_size` | `1` | Batch size for evaluation |
| `--max_samples` | `2` | Maximum number of samples to evaluate |
| `--temperature` | `0.0` | Generation temperature (0.0 = deterministic) |
| `--max_new_tokens` | `512` | Maximum tokens to generate |
| `--push_to_hub` | `False` | Push results to Hugging Face Hub |
| `--override_chat_template` | `False` | Override model's chat template |

## 📈 Understanding Results

The evaluation generates comprehensive results including:

### Metrics Computed
- **Exact Match (EM)**: Percentage of outputs that exactly match the expected response
- **ROUGE-1**: Overlap of unigrams between generated and expected text
- **ROUGE-2**: Overlap of bigrams between generated and expected text

You can also add your own metrics. 

### Output Structure
```
evaluation_results/
├── results/
│   └── [model_name]/
│       └── results_[timestamp].json
└── details/
    └── [model_name]/
        └── [timestamp]/
            └── details_[task]_[timestamp].parquet
```

### Sample Results
```json
{
  "results": {
    "custom|custom_task|0": {
      "em": 0.0,
      "em_stderr": 0.0,
      "rouge1": 0.002680965147453083,
      "rouge1_stderr": 0.0026809651474530827,
      "rouge2": 0.0,
      "rouge2_stderr": 0.0
    }
  }
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Errors**: Ensure the model name is correct and accessible
2. **CSV Format Issues**: Verify your CSV has the required `input` and `output` columns

## 📝 Example Datasets

The repository includes a sample dataset (`data/test.csv`) with 5 examples for testing:

```csv
input,output
"The weather is beautiful today and the sun is shining brightly.","I love sunny days when I can go for a walk in the park."
"Machine learning algorithms are becoming more sophisticated every year.","AI technology continues to advance rapidly with new breakthroughs."
```

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 🙏 Acknowledgments

- Built on top of the [Hugging Face LightEval](https://github.com/huggingface/lighteval) framework
- Uses the [Hugging Face Transformers](https://github.com/huggingface/transformers) library
- Inspired by the need for easy custom dataset evaluation using Lighteval
