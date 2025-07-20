import random
import sys

class PromptSampler:
    def __init__(self, prompt_filepath, num_samples=1):
        self.prompts = self._load_prompts(prompt_filepath)
        self.num_samples = num_samples # how many prompts to sample at a time
        if not self.prompts:
            raise ValueError("The file contains no valid prompts.")
    
    def _load_prompts(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        return prompts
    
    def sample_prompt(self, num_samples=None):
        """Randomly sample and return a prompt."""
        return random.choices(self.prompts, k=num_samples if num_samples is not None else self.num_samples)
