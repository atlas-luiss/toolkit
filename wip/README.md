# CTK — Collage Toolkit

A [Collage Lab](https://www.vincenzolomonaco.com/collage-lab) initiative.


## Structure

```
ctk/
  models/           model wrappers (ViT, CLIP, etc)
  strategies/       learning strategies 
  scenarios/        task splitting 
  benchmarks/       dataset loaders 
  trainers/         training loops 
  callbacks/        hooks: logging, checkpointing, metrics
  utils/            PEFT utilities

experiments/        vision experiment configs + SLURM scripts
examples/llm/       LLM fine-tuning examples

```
