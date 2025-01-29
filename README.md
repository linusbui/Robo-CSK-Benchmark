# RoboCSKBench: Benchmarking Embodied Commonsense Capabilities of Large Language Models

ToDo: Add Abstract

## Embodied Commonsense

ToDo: Add our definition

## Benchmark Overview

We propose **RoboCSKBench**, which is a language-driven benchmark for assessing the embodied commonsense capabilities of large language models as well as any approach working in the robotic household domain.
The benchmark currently focuses on the five most common commonsense aspects investigated by robotics researchers in the past ten years, as was described in [1].

We envision the proposed benchmark to be an extendable framework focusing not only on the five exemplary tasks we propose here, but also on future tasks to be incorporated under the *RoboCSKBench* umbrella, similar to the GLUE benchmark for natural language understanding [2].
We encourage other researchers and practitioners alike to benchmark their approaches and models, either on a single tasks or on the whole benchmark.

The five tasks currently included in the benchmark are the following:
- **[Tidy Up](./tidy_up/README.md)**: This task is centred around the problem of identifying prototypical locations for objects inside a household environment.
The main focus is on providing the model with a specific household object and returning a ranked list of locations in a household where the object can be expected.
The data can also be used to reason about objects that are out of place at their current location and thus need to be moved to a different place.
- **[Tool Usage](./tool_usage/README.md)**: The second task focuses on the concept of affordances, which describe what an object or the environment offers an agent [3].
We provide a list of objects with their respective affordances and a list of example tasks for each affordance. 
- **[Meta-Reasoning](./meta_reasoning/README.md)**: To evaluate the models capabilities in evaluating whether a specific robotic hardware setup is capable of executing a specific task, we propose this task.
In the data, we provide different tasks written as single sentence natural language instructions and a robotic setup that has was demonstrated to be capable of executing the task.
For each robotic setup, we provide the following information: Is the robot stationary or can it move? How many arms does it have? How many degrees of freedom does the arms have? What type of gripper is used? Are the grippers soft or rigid?
- **[Table Setting](./table_setting/README.md)**: In this task, the embodied agent should set the dinner table for a specific meal. 
This includes choosing the correct type of cutlery and the correct type of plate, so we provide a list of meals with their respective cutlery and plate.
- **[Cooking Procedures Knowledge](https://gitlab.ub.uni-bielefeld.de/s.kenneweg/temporal-cs)**: The last task evaluates a models ability to reason about the correct chronological sequence of two steps from a cooking recipe. 
Inspired by the *Event Ordering* task from [4], the model must decide whether on step is done before or after another.
Therefore, we provide the recipe title and two specific cooking steps from this recipe.

## Publication

The accompanying paper is currently under review.

## References

[1] J.-P. Töberg, A.-C. N. Ngomo, M. Beetz, and P. Cimiano, ‘Commonsense knowledge in cognitive robotics: a systematic literature review’, Front. Robot. AI, vol. 11, 2024, doi: 10.3389/frobt.2024.1328934.

[2] A. Wang, A. Singh, J. Michael, F. Hill, O. Levy, and S. R. Bowman, ‘GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding’, in ICLR 2019 · The Seventh International Conference on Learning Representations, New Orleans, LA, USA, 2019. doi: 10.48550/ARXIV.1804.07461.

[3] M. H. Bornstein and J. J. Gibson, ‘The Ecological Approach to Visual Perception’, The Journal of Aesthetics and Art Criticism, vol. 39, no. 2, p. 203, 1980, doi: 10.2307/429816.

[4] L. Zhang, Q. Lyu, and C. Callison-Burch, ‘Reasoning about Goals, Steps, and Temporal Ordering with WikiHow’, in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), Online: Association for Computational Linguistics, 2020, pp. 4630–4639. doi: 10.18653/v1/2020.emnlp-main.374.
