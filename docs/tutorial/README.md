# Tutorial

Continual learning step by step, from your data to your own method. Each page explains the idea
first, then shows code you can run as-is (seconds, on a CPU or GPU).

| step | page | you learn |
|---|---|---|
| 1 | [Your data](1_your_data.md) | what the toolkit needs; tensors, image folders, synthetic data |
| 2 | [Create the benchmark](2_benchmark.md) | split data into a sequence of tasks; class order, domains, few-shot |
| 3 | [Choose a scenario](3_scenario.md) | class-, task- and domain-incremental; reading the results |
| 4 | [LoRA, adapters and prompts](4_lora_adapters_prompts.md) | small trainable parts on a frozen pre-trained model |
| 5 | [Choose the classifier](5_classifier.md) | trained, statistical and correcting heads |
| 6 | [Routing](6_routing.md) | one expert per task and a router that picks it |
| 7 | [Your own CL method](7_your_method.md) | write a method from scratch: loss, memory, between-task steps, callbacks |
| 8 | [Write the CL loop yourself](8_your_loop.md) | train and evaluate step by step; act between them (e.g. merge models) |
| 9 | [Your own model](9_your_model.md) | any network, a model class, pre-trained models by name |

Just want to run something? [Getting started](../getting_started.md) is one page.
