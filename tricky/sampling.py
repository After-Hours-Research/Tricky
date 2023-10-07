import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from tqdm import tqdm

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizer, PreTrainedModel

from typing import Optional
from tricky.metrics import compute_perplexity


class PerplexitySampler:
    def __init__(
            self, 
            dataset: Dataset, 
            model: PreTrainedModel, 
            tokenizer: PreTrainedTokenizer, 
            batch_size: int = 32, 
            indices_to_ignore: Optional[list] = None
        ):
        logger.info("Initializing PerplexitySampler")
        
        self.dataset = dataset
        
        self.model = model
        self.model.eval()

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=False, return_tensors="pt"
        )

        if indices_to_ignore is not None:
            logger.info(f"Ignoring {len(indices_to_ignore)} indices")
            self.indices_to_ignore = indices_to_ignore

        unused_indicies = set(range(len(self.dataset))) - set(indices_to_ignore)

        self.dataloader = DataLoader(
            self.dataset[unused_indicies], 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=data_collator
        )

        self.indices_to_ignore = indices_to_ignore

        self._get_perplexities()
        self.model.eval(False)

    def _get_perplexities(self):
        logger.info("Calculating perplexities")
        perps = compute_perplexity(self.model, self.dataloader, aggregate=False)
        self.perplexities = np.array(perps)
        
    def distribution(self, K: int = 5, outlier_threshold: Optional[float] = None):
        logger.info(f"Plotting distribution with {K} bins")

        if outlier_threshold is not None:
            logger.info(f"Clipping perplexities to {outlier_threshold}")
            data_to_plot = np.clip(self.perplexities, 0, outlier_threshold)
        else:
            data_to_plot = self.perplexities
        
        sns.kdeplot(data_to_plot)
        
        # Compute and plot K-percentiles
        
        for edge in np.linspace(0, data_to_plot.max(), K - 1):
            plt.axvline(edge, color='r', linestyle='--')
        
        plt.xlabel('Perplexity')
        plt.title('Perplexity Distribution')
        plt.show()

    def _resolve_indices(self, selected_indices, indices_to_ignore):
        indices_to_ignore = sorted(indices_to_ignore)
        resolved_indices = []
        j = 0  # Pointer for indices_to_ignore
        offset = 0  # The current offset to be added to selected_indices

        for i in selected_indices:
            while j < len(indices_to_ignore) and indices_to_ignore[j] <= i + offset:
                offset += 1
                j += 1
            resolved_indices.append(i + offset)
            
        return resolved_indices
        
    def _sample_based_on_distribution(self, N: int, K: int, max_perplexity: Optional[float] = None):
        logger.info(f"Sampling {N} data points based on natural distribution across {K} bins")
        
        
        if max_perplexity is not None:
            logger.info(f"Clipping perplexities to {max_perplexity}")
            local_perplexities = np.clip(self.perplexities, 0, max_perplexity)
        else:
            local_perplexities = self.perplexities
        
        # add an extra bin for the max perplexity
        digitized = np.digitize(local_perplexities, np.linspace(min(local_perplexities), max(local_perplexities), K + 1))
        
        bin_counts = np.bincount(digitized)[1:K+1]
        bin_probs = bin_counts / np.sum(bin_counts)
        
        sampled_indices = []
        
        bin_idx = np.random.choice(range(1, K + 1), p=bin_probs, size=N)
        for i in tqdm(range(N), desc="Sampling based on distribution"):
            sampled_indices.append(np.random.choice(np.where(digitized == bin_idx[i])[0]))
        
        logger.info("Sampling complete.")
        return self._resolve_indices(sampled_indices, self.indices_to_ignore)

    def _sample_based_on_proportions(self, N: int, K: int, proportion_per_bin: dict, max_perplexity: Optional[float] = None):
        logger.info(f"Sampling {N} data points based on user-defined proportions across {K} bins")
        
        if max_perplexity is not None:
            logger.info(f"Clipping perplexities to {max_perplexity}")
            local_perplexities = np.clip(self.perplexities, 0, max_perplexity)
        else:
            local_perplexities = self.perplexities
        
        digitized = np.digitize(local_perplexities, np.linspace(min(local_perplexities), max(local_perplexities), K + 1))
        
        sampled_indices = []
        for k in tqdm(range(1, K + 1), desc="Sampling based on proportions"):
            proportion = proportion_per_bin.get(k, 0)
            n_samples = int(N * proportion)
            indices_in_bin = np.where(digitized == k)[0]
            if len(indices_in_bin) >= n_samples:
                sampled_indices.extend(np.random.choice(indices_in_bin, n_samples, replace=False))
            else:
                logger.warning(f"Bin {k} has {len(indices_in_bin)} samples, but {n_samples} were requested. Sampling with replacement.")
                sampled_indices.extend(np.random.choice(indices_in_bin, n_samples, replace=True))
        return self._resolve_indices(sampled_indices, self.indices_to_ignore)

    
    def sample(self, N: int, K: int = 5, proportion_per_bin: Optional[dict] = None, max_perplexity: float = None):
        if proportion_per_bin is None:
            indicies = self._sample_based_on_distribution(N, K, max_perplexity)
        else:
            indicies = self._sample_based_on_proportions(N, K, proportion_per_bin, max_perplexity)
        return self.dataset[indicies]
    
