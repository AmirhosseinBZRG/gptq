import time

import torch
import torch.nn as nn

from gptq import *
from modelutils import *
from quant import *
from scipy import stats
import numpy as np


def get_pythia(model_name):
    import torch
    
    # Skip initialization functions
    def skip(*args, **kwargs):
        pass
    
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    # Import the Pythia model class
    from transformers import AutoModelForCausalLM 
    
    # Load the Pythia model
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto')
    
    # Set the sequence length based on the model's configuration
    model.seqlen = model.config.max_position_embeddings  # Use max_position_embeddings for Pythia
    
    return model





@torch.no_grad()
def pythia_eval(model, testenc, dev, confidence_level=0.95):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.embed_positions = model.model.embed_positions.to(dev)
    if hasattr(model.model, 'project_out') and model.model.project_out:
        model.model.project_out = model.model.project_out.to(dev) 
    if hasattr(model.model, 'project_in') and model.model.project_in:
        model.model.project_in = model.model.project_in.to(dev) 
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.embed_positions = model.model.embed_positions.cpu()
    if hasattr(model.model, 'project_out') and model.model.project_out:
        model.model.project_out = model.model.project_out.cpu()
    if hasattr(model.model, 'project_in') and model.model.project_in:
        model.model.project_in = model.model.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.final_layer_norm is not None:
        model.model.final_layer_norm = model.model.final_layer_norm.to(dev)
    if model.model.project_out is not None:
        model.model.project_out = model.model.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.final_layer_norm is not None:
            hidden_states = model.model.final_layer_norm(hidden_states)
        if model.model.project_out is not None:
            hidden_states = model.model.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    # Calculate perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f'Perplexity: {ppl.item()}')

    # Calculate confidence interval
    nlls_np = np.array(nlls)
    mean_nll = np.mean(nlls_np)
    std_nll = np.std(nlls_np)
    n = len(nlls_np)

    # Calculate the z-score for the desired confidence level
    z_score = stats.norm.ppf((1 + confidence_level) / 2)

    # Calculate the confidence interval
    ci_lower = mean_nll - z_score * (std_nll / np.sqrt(n))
    ci_upper = mean_nll + z_score * (std_nll / np.sqrt(n))

    # Convert to perplexity scale
    ci_lower_ppl = torch.exp(torch.tensor(ci_lower) / model.seqlen)
    ci_upper_ppl = torch.exp(torch.tensor(ci_upper) / model.seqlen)

    print(f'Confidence Interval for Perplexity: [{ci_lower_ppl.item()}, {ci_upper_ppl.item()}]')

    model.config.use_cache = use_cache

def gpt_multigpu(model, gpus):
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(gpus[0])
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(gpus[0])
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(gpus[0])
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(gpus[-1])
    if hasattr(model.model.decoder, 'final_layer_norm') and model.model.decoder.final_layer_norm:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(gpus[-1])
    import copy
    model.lm_head = copy.deepcopy(model.lm_head).to(gpus[-1])

    cache = {'mask': None}

    class MoveModule(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.dev = next(iter(self.module.parameters())).device
        def forward(self, *inp, **kwargs):
            inp = list(inp)
            if inp[0].device != self.dev:
                inp[0] = inp[0].to(self.dev)
            if cache['mask'] is None or cache['mask'].device != self.dev:
                cache['mask'] = kwargs['attention_mask'].to(self.dev)
            kwargs['attention_mask'] = cache['mask']
            tmp = self.module(*inp, **kwargs)
            return tmp

    layers = model.model.decoder.layers
    pergpu = math.ceil(len(layers) / len(gpus))
    for i in range(len(layers)):
        layers[i] = MoveModule(layers[i].to(gpus[i // pergpu]))

    model.gpus = gpus


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='OPT model to load; pass `facebook/opt-X`.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    ) 
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--trits', action='store_true',
        help='Whether to use trits for quantization.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--load', type=str, default='',
        help='Load quantized model.'
    )
    parser.add_argument(
        '--benchmark', type=int, default=0,
        help='Number of tokens to use for benchmarking.'
    )
    parser.add_argument(
        '--check', action='store_true',
        help='Whether to compute perplexity during benchmarking for verification.'
    )
    parser.add_argument(
        '--eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(
        '--faster-kernel', action='store_true',
        help='Whether to use the new faster kernel for benchmarking.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--static-groups', action='store_true',
        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
    )

    args = parser.parse_args()
    
    if args.load:
        model = load_quant3(args.model, args.load)
    else:
        model = get_pythia(args.model)
        model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )


    if args.eval:
     datasets = ['wikitext2', 'hellaswag']
     for dataset in datasets:
         dataloader, testloader = get_loaders(dataset, seed=args.seed, model=args.model, seqlen=model.seqlen)
         print(dataset)
         pythia_eval(model, testloader, DEV)

    
