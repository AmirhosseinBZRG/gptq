import huggingface_hub
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import scipy
from scipy import stats
from torch.profiler import profile, record_function, ProfilerActivity

#import quant

#from gptq import GPTQ, Observer
from utils import find_layers, DEV, set_seed, get_wikitext2,  get_loaders, export_quant_table, gen_conditions
from texttable import Texttable


def get_GPT2(model):
    huggingface_hub.login("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")
    access_token = "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"
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
    model.seqlen = 2048
    return model



@torch.no_grad()
def GPT2_eval(model, testenc, dev):
    print('Evaluating ...')
    model = model.to(dev)
    testenc = testenc.input_ids.to(dev)
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
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
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    with profile(activities=[ProfilerActivity.CUDA,ProfilerActivity.CPU],
         record_shapes=True,
         profile_memory=True, with_stack=True) as prof:
      with record_function("model_inference"):

          for i in range(len(layers)):
              print(i)
              layer = layers[i].to(dev)

              if args.nearest:
                  subset = find_layers(layer)
                  for name in subset:
                      quantizer = quant.Quantizer()
                      quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)
                      W = subset[name].weight.data
                      quantizer.find_params(W, weight=True)
                      subset[name].weight.data = quantizer.quantize(W).to(next(iter(layer.parameters())).dtype)

              for j in range(nsamples):
                  outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
              layers[i] = layer.cpu()
              del layer
              torch.cuda.empty_cache()
              inps, outs = outs, inps
    #print profiler results
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Export chrome trace
    prof.export_chrome_trace("/content/llama_eval_trace.json")

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())
    
     # Calculate confidence interval
    nlls = torch.tensor(nlls)
    mean_nll = nlls.mean().item()
    std_nll = nlls.std().item()
    confidence_level = 0.95
    degrees_freedom = nsamples - 1
    confidence_interval = stats.t.interval(confidence_level, degrees_freedom, mean_nll, std_nll / np.sqrt(nsamples))
    lower_bound = torch.exp(torch.tensor(confidence_interval[0]) / model.seqlen).item()
    upper_bound = torch.exp(torch.tensor(confidence_interval[1]) / model.seqlen).item()
    print(f"95% Confidence Interval for Perplexity: [{lower_bound}, {upper_bound}]")

    model.config.use_cache = use_cache



def GPT2_multigpu(model, gpus, gpu_dist):
    model.model.embed_tokens = model.model.embed_tokens.to(gpus[0])
    if hasattr(model.model, 'norm') and model.model.norm:
        model.model.norm = model.model.norm.to(gpus[0])
    import copy
    model.lm_head = copy.deepcopy(model.lm_head).to(gpus[0])

    cache = {'mask': None, 'position_ids': None}

    class MoveModule(nn.Module):

        def __init__(self, module, invalidate_cache):
            super().__init__()
            self.module = module
            self.dev = next(iter(self.module.parameters())).device
            self.invalidate_cache=invalidate_cache

        def forward(self, *inp, **kwargs):
            inp = list(inp)
            if inp[0].device != self.dev:
                inp[0] = inp[0].to(self.dev)

            if cache['mask'] is None or cache['mask'].device != self.dev or self.invalidate_cache:
                cache['mask'] = kwargs['attention_mask'].to(self.dev)
            kwargs['attention_mask'] = cache['mask']

            if cache['position_ids'] is None or cache['position_ids'].device != self.dev or self.invalidate_cache:
                cache['position_ids'] = kwargs['position_ids'].to(self.dev)
            kwargs['position_ids'] = cache['position_ids']
            
            tmp = self.module(*inp, **kwargs)
            return tmp

    layers = model.model.layers
    from math import ceil
    if not gpu_dist:
        pergpu = ceil(len(layers) / len(gpus))
        for i in range(len(layers)):
            layers[i] = MoveModule(layers[i].to(0 if i == 0 or i == len(layers) -1 else gpus[(i-1) // pergpu]), i==0)
    else:
        assert gpu_dist[0] >= 2, "At least two layers must be on GPU 0."
        assigned_gpus = [0] * (gpu_dist[0]-1)
        for i in range(1, len(gpu_dist)):
            assigned_gpus = assigned_gpus + [i] * gpu_dist[i]

        remaining_assignments = len(layers)-len(assigned_gpus) - 1
        if remaining_assignments > 0:
            assigned_gpus = assigned_gpus + [-1] * remaining_assignments

        assigned_gpus = assigned_gpus + [0]

        for i in range(len(layers)):
            layers[i] = MoveModule(layers[i].to(gpus[assigned_gpus[i]]), i==0)

    model.gpus = gpus



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str, help='llama model to load')
    parser.add_argument('dataset', type=str, choices=['wikitext2', 'hellaswag'], help='Where to extract calibration data from.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration data samples.')
    parser.add_argument('--percdamp', type=float, default=.01, help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--nearest', action='store_true', help='Whether to run the RTN baseline.')
    parser.add_argument('--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16], help='#bits to use for quantization; use 16 for evaluating base model.')
    parser.add_argument('--trits', action='store_true', help='Whether to use trits for quantization.')
    parser.add_argument('--groupsize', type=int, default=-1, help='Groupsize to use for quantization; default uses full row.')
    parser.add_argument('--eval', action='store_true', help='evaluate quantized model.')
    parser.add_argument('--test-generation', action='store_true', help='test generation.')
    parser.add_argument('--save', type=str, default='', help='Save quantized checkpoint under this name.')
    parser.add_argument('--save_safetensors', type=str, default='', help='Save quantized `.safetensors` checkpoint under this name.')
    parser.add_argument('--load', type=str, default='', help='Load quantized model.')
    parser.add_argument('--benchmark', type=int, default=0, help='Number of tokens to use for benchmarking.')
    parser.add_argument('--check', action='store_true', help='Whether to compute perplexity during benchmarking for verification.')
    parser.add_argument('--sym', action='store_true', help='Whether to perform symmetric quantization.')
    parser.add_argument('--act-order', action='store_true', help='Whether to apply the activation order GPTQ heuristic')
    parser.add_argument('--true-sequential', action='store_true', help='Whether to run in true sequential model.')
    parser.add_argument('--new-eval', action='store_true', help='Whether to use the new PTB and C4 eval')
    parser.add_argument('--layers-dist', type=str, default='', help='Distribution of layers across GPUs. e.g. 2:1:1 for 2 layers on GPU 0, 1 layer on GPU 1, and 1 layer on GPU 2. Any remaining layers will be assigned to your last GPU.')
    parser.add_argument('--observe',
                        action='store_true',
                        help='Auto upgrade layer precision to higher precision, for example int2 to int4, groupsize 128 to 64. \
            When this feature enabled, `--save` or `--save_safetensors` would be disable.')
    parser.add_argument('--quant-directory', type=str, default=None, help='Specify the directory for export quantization parameters to toml format. `None` means no export by default.')

    args = parser.parse_args()
    if args.load:
        model = load_quant3(args.model, args.load)
    else:
        model = get_opt(args.model)
        model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    if args.wbits < 16 and not args.nearest:
        tick = time.time()
        quantizers = opt_sequential(model, dataloader, DEV)
        print(time.time() - tick)

    if args.benchmark:
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            opt_multigpu(model, gpus)
        else:
            model = model.to(DEV)
        if args.benchmark:
            input_ids = next(iter(dataloader))[0][:, :args.benchmark]
            benchmark(model, input_ids, check=args.check)
    if args.load:
        exit()



    if args.eval:
     datasets = ['wikitext2', 'hellaswag']
     for dataset in datasets:
         dataloader, testloader = get_loaders(dataset, seed=args.seed, model=args.model, seqlen=model.seqlen)
         print(dataset)
         GPT2_eval(model, testloader, DEV)

    


