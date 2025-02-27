# !/usr/bin/env python3

"""
GRPO demo. (The basic framework base on the https://github.com/HarderThenHarder/transformers_tasks/tree/main/RLHF).

Replace the previous PPO with GRPO.

Author: Xinchen Han 
Date: 2025/02/27.
"""

import os
import time
import random
from rich import box
from rich.table import Table
from rich.align import Align
from rich.console import Console

import torch
import torch.nn.functional as F
from trl.grpo import GRPOTrainer
from trl.gpt2 import GPT2HeadWithValueModel
from transformers import AutoTokenizer
# from transformers import top_k_top_p_filtering

from iTrainingLogger import iSummaryWriter
from torch import Tensor

# import transformers
# print(transformers.__version__)


MODEL_CONFIG = {
    'model_name': 'uer/gpt2-chinese-cluecorpussmall',
    'device': 'cuda:0'
}
MIN_REWARD = -2.0
MAX_REWARD = 2.0

GROUP_NUM = 2 

LOG_PATH = './logs'
LOG_NAME = 'Terminal-Human-Feedback'
writer = iSummaryWriter(log_path=LOG_PATH, log_name=LOG_NAME)

prompts = [
            '刚收到货，感觉',
            '这部电影很',
            '说实话，真的很',
            '这次购物总的来说体验很'
        ]


def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def respond_to_batch(model, queries, txt_len=20, top_k=0, top_p=1.0):
    """
    根据prompt生成答案。

    Args:
        model (_type_): _description_
        queries (_type_): _description_
        txt_len (int, optional): _description_. Defaults to 20.
        top_k (int, optional): _description_. Defaults to 0.
        top_p (float, optional): _description_. Defaults to 1.0.

    Returns:
        _type_: _description_
    """
    input_ids = queries
    device = MODEL_CONFIG['device']
    for _ in range(txt_len):
        outputs = model(input_ids.to(device))
        next_token_logits = outputs[0][:, -1, :]
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1) # Return Index, not value.
        input_ids = torch.cat([input_ids.to(device), next_token.unsqueeze(-1)], dim=-1).cpu()
    return input_ids[:, -txt_len:]


def main():
    """
    主函数。
    """
    console = Console()
    table = Table(show_footer=False)
    table.width = console.width
    table.box = box.SQUARE
    table.row_styles = ["none", "dim"]
    console.clear()

    # add title
    table.title = (
            "[bold not italic]:robot:[/] Reinforcemnet Learning from Human Feedback - Terminal"
        )

    # add column (first line)
    table.add_column("config/key", no_wrap=True)
    table.add_column("config/value", no_wrap=True)
    
    # add config row to table
    for k, v in MODEL_CONFIG.items():
        table.add_row(k, v)
    table.add_row('log path', os.path.join(LOG_PATH, LOG_NAME))
    table.add_row('min ~ max reward', f'{MIN_REWARD} ~ {MAX_REWARD}')
    table.add_row('prompts', f'{prompts}')
    table.caption = "You can change config in [b not dim]Source Code[/]"
    
    table.columns[0].style = "bright_red"
    table.columns[0].header_style = "bold bright_red"
    table.columns[1].style = "bright_green"
    table.columns[1].header_style = "bold bright_green"
    table_centered = Align.center(table)
    console.print(table_centered)

    with console.status("[bold bright_green]Initializing Model & Env..."):
        model = GPT2HeadWithValueModel.from_pretrained(MODEL_CONFIG['model_name']).to(MODEL_CONFIG['device'])
        ref_model = GPT2HeadWithValueModel.from_pretrained(MODEL_CONFIG['model_name']).to(MODEL_CONFIG['device'])
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['model_name'])
        grpo_config = {'batch_size': GROUP_NUM, 'forward_batch_size': GROUP_NUM}
        grpo_trainer = GRPOTrainer(
            model,
            ref_model,
            tokenizer,
            **grpo_config
        )
        
        console.log('[bold magenta][Done] Initialized Model & Env.')

    step = 1
    t = time.time()

    reward_list = []
    query_list = []
    response_list = []

    while True:
        reward_list, query_list, response_list = [], [], []
        current_prompt = random.choice(prompts)
        console.print(f'[Step {step}]')
        console.print(f'[bright_yellow]prompt>>> {current_prompt}[/bright_yellow]')
        query_tensor = tokenizer.encode(current_prompt, return_tensors="pt").to(MODEL_CONFIG['device'])
        
        for i in range(GROUP_NUM):
            console.print(f'[Group {i+1} / {GROUP_NUM}]')
            console.print('generating results...', end='\r')
            response_tensor = respond_to_batch(model, query_tensor)
            response_txt = tokenizer.decode(response_tensor[0, :].to('cpu'))
            console.print(f'[bright_blue]result>>> {response_txt}[/bright_blue]')
            reward_txt = input(f'Reward ({MIN_REWARD} ~ {MAX_REWARD}): ')
            try:
                reward_f = float(reward_txt)
                if MIN_REWARD <= reward_f <= MAX_REWARD:
                    pass
                else:
                    reward_txt = input(f'Reward ({MIN_REWARD} ~ {MAX_REWARD}): ')
                    reward_f = float(reward_txt)
            except:
                reward_txt = input(f'Reward ({MIN_REWARD} ~ {MAX_REWARD}): ')
                reward_f = float(reward_txt)
            
            reward_list.append(reward_f)
            query_list.append(query_tensor)
            response_list.append(response_tensor)

        rewards = torch.tensor(reward_list, dtype=torch.float16).to(MODEL_CONFIG['device'])
        querys = torch.stack(query_list).to(MODEL_CONFIG['device'])  
        responses = torch.stack(response_list).to(MODEL_CONFIG['device'])  # Stack: Only every length of response[i] is same. 

        with console.status("[bold bright_green]Updating Model..."):
            # querys:[Group, 1, query_len], responses:[Group, 1, response_len], rewards:[group]
            grpo_trainer.step(querys, responses, rewards)
        
        writer.add_scalar('reward history', reward_f, step)
        writer.add_scalar('label time used', time.time() - t, step)
        writer.record()
        t = time.time()
        step += 1


if __name__ == '__main__':
    main()
