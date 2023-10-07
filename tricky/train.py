from transformers import TrainerCallback
from trl import SFTTrainer
from tricky.sampling import PerplexitySampler
from torch.utils.data import ConcatDataset, Subset
import torch
from datasets import Dataset
from loguru import logger

class CurriculumTrainer(SFTTrainer):
    def __init__(
            self, 
            n_phases: int = 5, 
            recompute_perplexities: bool = True, 
            *args, 
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.main_dataset = self.train_dataset
        self.n_phases = n_phases
        self.new_data_amount = len(self.main_dataset) // n_phases
        self.selected_indices = []
        self.nonselected_indices = list(range(len(self.main_dataset)))
        self.recompute_perplexities = recompute_perplexities
        self.initial_perplexity_sampler = None
        self.first_run = True
        self.current_dataloader = self.get_train_dataloader()
        self.current_phase = 0

    def get_train_dataloader(self):
        if self.first_run or self.recompute_perplexities:
            self.first_run = False
            sampler = PerplexitySampler(
                dataset=self.main_dataset, 
                model=self.model,
                tokenizer=self.tokenizer,
                batch_size=self.args.train_batch_size,
                indices_to_ignore=self.selected_indices
            )
            if self.initial_perplexity_sampler is None:
                self.initial_perplexity_sampler = sampler

        else:
            sampler = self.initial_perplexity_sampler
        

        self.current_phase += 1
        new_indices = sampler.sample(
            N=self.new_data_amount, 
            K=3, 
            proportion_per_bin={0:0, 1:1, 2:0},
            max_perplexity=1000
        )
        
        self.selected_indices.extend(new_indices)
        self.nonselected_indices = [i for i in self.nonselected_indices if i not in new_indices]
        combined_dataset = ConcatDataset([self.current_dataloader.dataset, Subset(self.main_dataset, new_indices)])
        self.train_dataset = combined_dataset
        return super().get_train_dataloader()

    def switch_phase(self):
        if self.current_phase >= self.n_phases:
            logger.info("Training complete")
            self.control.should_training_stop = True
        else:
            logger.info(f"Switching to phase {self.current_phase}")
            self.current_dataloader = self.get_train_dataloader()

class PerplexityStabilityCallback(TrainerCallback):
    def __init__(self, window_size=3, threshold=0.1, trainer: Trainer = None):
        self.window_size = window_size
        self.threshold = threshold
        self.past_perplexities = []
        self.trainer = trainer

    def scale_perplexity(self, perplexity, min_perplexity, max_perplexity):
        return (perplexity - min_perplexity) / (max_perplexity - min_perplexity)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            current_perplexity = torch.exp(metrics["eval_loss"])
            self.past_perplexities.append(current_perplexity)

            if len(self.past_perplexities) > self.window_size:
                self.past_perplexities.pop(0)
                
                max_perplexity = max(self.past_perplexities)
                min_perplexity = min(self.past_perplexities)
                scaled_perplexities = [self.scale_perplexity(p, min_perplexity, max_perplexity) for p in self.past_perplexities]

                if max(scaled_perplexities) - min(scaled_perplexities) < self.threshold:
                    self.trainer.switch_phase()
                    self.past_perplexities.clear()