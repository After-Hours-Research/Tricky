from transformers import PreTrainedModel
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm
import torch
import torch.nn.functional as F

def nonzero_avg(tensor: torch.tensor) -> torch.tensor:
    non_zero_sum = torch.sum(tensor, dim=1)
    non_zero_count = torch.count_nonzero(tensor, dim=1).float()
    avg = torch.zeros_like(non_zero_sum, dtype=torch.float32)
    mask = non_zero_count != 0
    avg[mask] = non_zero_sum[mask] / non_zero_count[mask]
    return avg

def compute_perplexity(
    model: PreTrainedModel, 
    instr_dataloader: DataLoader,
    aggregate: bool = True
) -> float:
    losses = []
    with torch.no_grad():
        for batch in tqdm(instr_dataloader, desc="Calculating Perplexity"):
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            if aggregate:
                losses.append(outputs.loss.item())
            else:
                logits = outputs.logits
                labels = batch["labels"]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none')
                loss = loss.view(labels.size(0), -1)
                loss = nonzero_avg(loss) # mean will be slightly different to global mean (above) - as this is a per sample mean
                losses.extend(loss.tolist())
    losses = torch.tensor(losses)
    if aggregate:
        return torch.exp(losses.mean()).item()
    else:
        return torch.exp(losses).tolist()