import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import PythiaForCausalLM, PythiaTokenizerFast
from datasets import load_dataset

def batched_perplexity(model, dataset, tokenizer, batch_size, stride):
    device = model.device
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    text_len = encodings.input_ids.size(1)
    lls = []

    for i in tqdm(range(0, text_len, batch_size * stride)):
        begin_locs, end_locs, trg_lens = [], [], []
        for j in range(batch_size):
            j = i + j * stride
            if j >= text_len:
                break
            begin_loc = max(j + stride - model.config.n_positions, 0)
            end_loc = min(j + stride, text_len)
            trg_len = end_loc - j  # may be different from stride on last loop

            begin_locs.append(begin_loc)
            end_locs.append(end_loc)
            trg_lens.append(trg_len)

        input_ids = [encodings.input_ids[:, b:e] for b, e in zip(begin_locs, end_locs)]
        target_end_locs = [sen.size(-1) for sen in input_ids]
        input_ids = [
            F.pad(sen, (0, model.config.n_positions - sen.size(-1)), "constant", 0) for sen in input_ids
        ]  # Padding to the max length
        input_ids = torch.stack(input_ids, dim=1).squeeze(0).to(device)

        target_ids = torch.ones_like(input_ids) * -100  # -100 is the default ignore_index value in CrossEntropyLoss
        for i, (b, e) in enumerate(zip(trg_lens, target_end_locs)):
            labels = input_ids[i, -b:e].clone()
            target_ids[i, -b:e] = labels

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs.loss * sum(trg_lens)

        lls.append(log_likelihood)

    ppl = torch.exp(sum(torch.stack(lls) / end_locs[-1]))
    return ppl

if __name__ == "__main__":
    device = "cuda"
    model_id = "your-pythia-model-id"  # Replace with your Pythia model ID
    model = PythiaForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = PythiaTokenizerFast.from_pretrained(model_id)
    max_len = model.config.n_positions
    stride = 512
    batch_size = 16

    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:100%]")
    ppl = batched_perplexity(model, test, tokenizer, batch_size, stride)
    print(f"--------------{ppl=}-------------")
