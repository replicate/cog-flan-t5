from typing import List, Optional, Union
from cog import BasePredictor, Input
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

CACHE_DIR = 'weights'
SEP = "<sep>"

class Predictor(BasePredictor):
    def setup(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", cache_dir=CACHE_DIR, local_files_only=True)
        self.model.to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl", cache_dir=CACHE_DIR, local_files_only=True)

    def predict(
        self,
        prompt: Union[str, List[str]] = Input(description=f"Prompt (or a list of prompts) to send to FLAN-T5."),
        n: int = Input(description="Number of output sequences to generate", default=1, ge=1, le=5),
        max_length: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            ge=1,
            default=50
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
            ge=0.01,
            le=5,
            default=0.75,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.01,
            le=1.0,
            default=1.0
        ),
        repetition_penalty: float = Input(
            description="Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, less than 1 encourage it.",
            ge=0.01,
            le=5,
            default=1
        )
        ) -> List[str]:
        input = self.tokenizer(prompt, return_tensors="pt", padding=True).input_ids.to(self.device)

        outputs = self.model.generate(
            input,
            num_return_sequences=n,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
        out = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return out
        