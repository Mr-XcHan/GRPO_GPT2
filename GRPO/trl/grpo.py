# Cell
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
import torch
import collections
import time
import random

from transformers import DataCollatorForLanguageModeling

from .core import (logprobs_from_logits,
                      whiten,
                      clip_by_value,
                      entropy_from_logits,
                      flatten_dict,
                      average_torch_dicts,
                      stats_to_np,
                      stack_dicts,
                      add_suffix,
                      WANDB_PADDING)

# Cell

class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """
    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult

# Cell

class FixedKLController:
    """Fixed KL controller."""
    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass

# Cell

class GRPOTrainer:
    """
    The GRPO_trainer uses  Group Relative Policy Optimization to optimise language models.
    """

    default_params = {
        "lr": 1.41e-5,
        "adap_kl_ctrl": True,
        "init_kl_coef":0.2,
        "target": 6,
        "horizon":10000,
        "gamma":1,
        "lam":0.95,
        "cliprange": .2,
        "cliprange_value":.2,
        "vf_coef":.1,
        "batch_size": 256,
        "forward_batch_size": 16,
        "grpo_epochs": 4,
    }

    def __init__(self, model, ref_model, tokenizer, **grpo_params):
        """
        Initialize GRPO Trainer.
        
        """
        self.grpo_params = self.default_params
        self.grpo_params.update(grpo_params)

        self.ref_model = ref_model
        self.model = model
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

        self.optimizer = Adam(model.parameters(), lr=self.grpo_params['lr'])

        if self.grpo_params['adap_kl_ctrl']:
            self.kl_ctl = AdaptiveKLController(self.grpo_params['init_kl_coef'],
                                               self.grpo_params['target'],
                                               self.grpo_params['horizon'])
        else:
            self.kl_ctl = FixedKLController(self.grpo_params['init_kl_coef'])


    def step(self, queries, responses, scores):
        """
        Run a GRPO optimisation step.

        args:
            queries (List): List of tensors containing the encoded queries, shape [query_length]
            responses (List): List of tensors containing the encoded responses, shape [response_length]
            scores (List): tensor containing the scores, shape [batch_size]

        returns:
            train_stats (dict): a summary of the training statistics
        """

        bs = self.grpo_params['batch_size']  # group_len
        assert bs == len(queries), f"Batch size ({bs}) does not match number of examples ({len(queries)})"

        timing = dict()
        t0 = time.time()

        response_lengths = [len(r[0]) for r in responses]

        t = time.time()
        logprobs, ref_logprobs = self.batched_forward_pass(queries, responses)          # 拿到模型生成的tokens的log_prob、token_value
        timing['time/grpo/forward_pass'] = time.time()-t

        t = time.time()
        rewards = self.compute_rewards(scores, logprobs, ref_logprobs)        # 计算discount reward
        timing['time/grpo/compute_rewards'] = time.time()-t

        t = time.time()
        all_stats = []
        idxs = list(range(bs))
        for _ in range(self.grpo_params['grpo_epochs']):
            train_stats = self.train_minibatch(logprobs, ref_logprobs,rewards, queries, responses, torch.cat((queries,responses), dim=-1))
            all_stats.append(train_stats)
        timing['time/grpo/optimize_step'] = time.time()-t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats['policy/advantages'] = torch.flatten(train_stats['policy/advantages']).unsqueeze(0)
        train_stats['policy/advantages'] = torch.nan_to_num(train_stats['policy/advantages'], WANDB_PADDING)
        train_stats['policy/ratio'] = torch.flatten(train_stats['policy/ratio']).unsqueeze(0)

        stats = self.record_step_stats(scores=scores, logprobs=logprobs, ref_logprobs=ref_logprobs,
                                       train_stats=train_stats, kl_coef=self.kl_ctl.value)
        stats = stats_to_np(stats)
        timing['time/grpo/calc_stats'] = time.time()-t

        timing['time/grpo/total'] = time.time()-t0
        stats.update(timing)
        return stats

    def batched_forward_pass(self, queries, responses):
        """Calculate model outputs in multiple batches."""
        bs = self.grpo_params['batch_size']
        fbs = self.grpo_params['forward_batch_size']
        all_logprobs = []
        all_ref_logprobs = []

        for i in range(int(bs/fbs)):
            query_batch = queries[i*fbs:(i+1)*fbs].squeeze(1) # query_batch:[group, query_len]
            response_batch = responses[i*fbs:(i+1)*fbs].squeeze(1) # reponse_batch:[group, response_len]
            input_ids = self.data_collator([torch.cat([q, r]) for q, r in zip(query_batch, response_batch)])["input_ids"]
            with torch.no_grad():
                logits, _, _ = self.model(input_ids)  # Don't need Value. logits -> (batch, seq_len, vocab_size)
                ref_logits, _, _ = self.ref_model(input_ids) # Same shape with logits.
            logprobs = logprobs_from_logits(logits[:,:-1,:], input_ids[:,1:])           # (batch, seq_len - 1)
            ref_logprobs = logprobs_from_logits(ref_logits[:,:-1,:], input_ids[:,1:])   # (batch, seq_len - 1)
            for j in range(fbs):
                start = len(query_batch[j]) - 1                                         # 拿到模型生成部分的信息（去掉prompt部分的信息）
                end = start + len(response_batch[j])
                all_logprobs.append(logprobs[j, start:end])                             # 生成的tokens的概率
                all_ref_logprobs.append(ref_logprobs[j, start:end])                     # ref model生成的tokens的概率
        
        all_logprobs = torch.stack(all_logprobs) # [batch, response_len]
        all_ref_logprobs = torch.stack(all_ref_logprobs) # [batch, response_len]
        return all_logprobs, all_ref_logprobs

    def train_minibatch(self, logprobs, ref_logprobs, rewards, query, response, model_input):
        """Train one GRPO minibatch"""
        loss_p, train_stats  = self.loss(logprobs, ref_logprobs,rewards, query, response, model_input)
        self.optimizer.zero_grad()
        loss_p.backward()
        self.optimizer.step()
        return train_stats

    def compute_rewards(self, scores, logprobs, ref_logprobs):
        """Compute per token rewards from scores and KL-penalty."""
        rewards= []
        for score, logprob, ref_logprob in zip(scores, logprobs, ref_logprobs):
            kl = torch.exp(ref_logprob - logprob) - (ref_logprob - logprob) - 1  # Eq.(4)
            kl = torch.cumsum(kl, dim=-1)  # REINFORCE ++ 
            reward = score - kl             
            reward[-1] += score                                                       
            rewards.append(reward)
        return torch.stack(rewards).squeeze(1) # (batch, response_len)

    def loss(self, old_logprobs, ref_logprobs, rewards, query, response, model_input):
        """Calculate policy loss"""
        lastgaelam = 0
        advantages_reversed = []
        gen_len = response.shape[-1]

        group_mean = torch.mean(rewards, dim=0, keepdim=True) # [1, seq_len]
        group_std = torch.std(rewards, dim=0, keepdim=True) # [1, seq_len]

        rewards = (rewards - group_mean) / (group_std + 1e-8)
        advantages = torch.cumsum(rewards, dim=-1)  # Process Supervision.
        advantages = whiten(advantages)
        advantages = advantages.detach() # [group, seq_len]

        logits, _, _ = self.model(model_input)
        logits = logits.squeeze() # logits -> (batch, all_seq_len, vocab_size)
        model_input = model_input.squeeze()
        logprobs = logprobs_from_logits(logits[:,:-1,:], model_input[:, 1:])

        #only the generation part of the values/logprobs is needed
        logprobs = logprobs[:, -gen_len:] # logprob -> (batch, generated_seq_len)
        old_logprobs = old_logprobs[:, -gen_len:] # (batch, generated_seq_len)

        ratio = torch.exp(logprobs - old_logprobs)
        pg_losses = -advantages * ratio  # importance sampling [batch, gen_len]
        pg_losses2 = -advantages * torch.clamp(ratio,
                                               1.0 - self.grpo_params['cliprange'],
                                               1.0 + self.grpo_params['cliprange'])

        pg_loss = torch.mean(torch.max(pg_losses, pg_losses2))
        pg_clipfrac = torch.mean(torch.gt(pg_losses2, pg_losses).double())

        KL = torch.mean(torch.exp(ref_logprobs - logprobs) - (ref_logprobs - logprobs) - 1)
        pg_loss = pg_loss - KL

        loss = pg_loss 

        entropy = torch.mean(entropy_from_logits(logits))
        approxkl = .5 * torch.mean((logprobs - old_logprobs)**2)
        policykl = torch.mean(logprobs - old_logprobs)
        # return_mean, return_var = torch.mean(returns), torch.var(returns)
        # value_mean, value_var = torch.mean(values), torch.var(values)

        stats = dict(
            loss=dict(policy=pg_loss, total=loss),
            policy=dict(entropy=entropy, approxkl=approxkl,policykl=policykl, advantages=advantages, advantages_mean=torch.mean(advantages), ratio=ratio),        )
        return pg_loss, flatten_dict(stats)


    def record_step_stats(self, kl_coef, **data):
        """Record training step statistics."""
        kl_list = [logprobs-ref_logprobs for logprobs, ref_logprobs in zip(data['logprobs'], data['ref_logprobs'])]
        mean_kl = torch.mean(torch.stack([torch.sum(kl) for kl in kl_list]))
        mean_entropy = torch.mean(torch.stack([torch.sum(-log_probs) for log_probs in data['logprobs']]))
        # mean_non_score_reward =torch.mean(torch.stack([torch.sum(non_score_reward) for non_score_reward in data['non_score_reward']]))
        stats = {
            'objective/kl': mean_kl,
            'objective/kl_dist': kl_list,
            'objective/logprobs': data['logprobs'],
            'objective/ref_logprobs': data['ref_logprobs'],
            'objective/kl_coef': kl_coef,
            'objective/entropy': mean_entropy,
        }

        for k, v in data['train_stats'].items():
            stats[f'grpo/{k}'] = torch.mean(v, axis=0)
        return stats
