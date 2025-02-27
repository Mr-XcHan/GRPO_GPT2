# A GRPO DMEO BASED ON GPT2.

A reproduction of grpo from deepseek based on GPT2.

The basic framework is cited from: https://github.com/HarderThenHarder/transformers_tasks/tree/main/RLHF

I added the grpo.py ( from DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models: https://arxiv.org/abs/2402.03300):

    1. The Group parameter can be user-defined.
    
    2. Process Supervision RL with GRPO, rather than Outcome Supervision RL with GRPO.
    
    3. Token reward is defined as Eq.(7) in REINFORCE++ (https://arxiv.org/abs/2501.03262).
