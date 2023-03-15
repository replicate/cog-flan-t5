from typing import Any, Iterator, List, Optional
from cog import BasePredictor, Input
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.generation.utils import GenerationMixin
import torch
from patch_sample import yield_sample

CACHE_DIR = 'weights'

# Monkey patch the sample function to use yield_sample for gradual decoding
stream_sample = yield_sample
no_stream_sample = GenerationMixin.sample

class Predictor(BasePredictor):
    def setup(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", cache_dir=CACHE_DIR, local_files_only=True)
        self.model.eval()
        self.model.to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl", cache_dir=CACHE_DIR, local_files_only=True)
        self.tokens_to_scrub = set((self.tokenizer.bos_token, self.tokenizer.eos_token, self.tokenizer.pad_token, self.tokenizer.unk_token))

    def predict(
        self,
        prompt: str = Input(description=f"Prompt to send to FLAN-T5."),
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
            default=1,),
        stream: bool = Input(description="Return a progressive stream of output tokens. Useful for longer inputs or interactive conversations", default=True)
        ) -> Iterator[Any]:
        print('generating')
        input = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        with torch.inference_mode():
            if stream:
                T5ForConditionalGeneration.sample = stream_sample
                for output in self.model.generate(
                    input,
                    num_return_sequences=n,
                    max_length=max_length,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty
                ):
                    out = self.tokenizer.batch_decode(output, skip_special_tokens=True)
                    yield [val if val not in self.tokens_to_scrub else "" for val in out]
            else:
                T5ForConditionalGeneration.sample = no_stream_sample
                output = self.model.generate(
                    input,
                    num_return_sequences=n,
                    max_length=max_length,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty
                )
                out = self.tokenizer.batch_decode(output, skip_special_tokens=True)
                return out

if __name__ == '__main__':
    p = Predictor()
    p.setup()
    p_one = p.predict("The following is a list of", n=3, max_length=50, temperature=0.75, top_p=1.0, repetition_penalty=1.0, stream=False)
    print(p_one)
    for out in p.predict("The following is a list of", n=3, max_length=50, temperature=0.75, top_p=1.0, repetition_penalty=1.0, stream=True):
        print(out)
        