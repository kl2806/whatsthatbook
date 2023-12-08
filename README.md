# whatsthatbook

This repository contains code for the paper [Decomposing Complex Queries for Tip-of-the-tongue Retrieval](https://arxiv.org/abs/2305.15053)


## Data
Dataset is released on huggingface at: https://huggingface.co/datasets/nlpkevinl/whatsthatbook

## Installation

Create conda environment

``` sh
conda create -n whatsthatbook python=3.9 --yes
conda activate whatsthatbook
```

Install the requirements.txt file

``` sh
pip install requirements.txt
```

## Clue Generation

To generate subqueries, input a prompt, examples, and the original query file. Our example uses `openai` as the LLM decomposer, so first set the required environmental variables `OPENAI_API_KEY`.

Example query for cover prompts. 

```
python clue_extraction.py --prompt_text_file prompts/cover.prompt.txt --examples_file prompts/examples/cover_clue_examples.jsonl  --input_file ./data/2022-05-30_14441_gold_posts.cover_clues.jsonl --output_file ./data/debug.2022-05-30_14441_gold_posts.cover_clues.jsonl  --max_examples 1

```


## References

Please consider referencing our work if you find it useful for your work

```
@misc{lin-etal:2023:arxiv,
  author    = {Kevin Lin and Kyle Lo and Joseph Gonzalez and Dan Klein},
  title     = {Decomposing Complex Queries for Tip-of-the-tongue Retrieval},
  note      = {arXiv:2305.15053},
  year      = {2023}
}
```