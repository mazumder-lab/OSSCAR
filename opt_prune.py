import time

import torch
import torch.nn as nn

import json

from modelutils import *
from osscar_prune import *

import numpy as np
import os



def get_opt(model, path, cached = True):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model, torch_dtype='auto', cache_dir=path, local_files_only=cached)
    model.seqlen = model.config.max_position_embeddings
    return model


@torch.no_grad()
def opt_sequential(model, dataloader, dev, num_heads, nsamples=128, parallel = False):
    print('Starting ...')
    dev2 = 'cpu'

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    seqlen = model.seqlen

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev) 
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev2
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.to(dev2)
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    
    
    if args.fullseq:
        inps2 = torch.clone(inps)
        outs2 = torch.zeros_like(inps)


    attention_mask = cache['attention_mask']

    print('Ready.')


    for i in range(len(layers)):
        layer = layers[i].to(dev)
        
        
        subset = find_layers(layer)
        osscar = {}
        print('----')
        print("name is ",list(subset.keys()))
        for name in subset:
            if name not in ['self_attn.out_proj','fc2']:
                continue
                
            osscar[name] = OSSCAR_prune(subset[name], args.algo, nsamples=nsamples, seqlen=seqlen, update_iter=args.update_iter,update_iter2=args.update_iter2, lambda2=args.lambda2, layername = name, num_heads = num_heads, local_out=args.local_out, local_fc=args.local_fc, local_iter=args.local_iter, local_test=args.local_test,fullseq = args.fullseq)

        def add_batch(name):
            def tmp(_, inp, out):
                osscar[name].add_batch(inp[0].data, out.data)
            return tmp
        
        

        
        if args.fullseq:
            # generate XTY
            print("DIFF is", torch.sum(torch.abs(inps2.to(torch.float64)-inps.to(torch.float64))) / torch.sum(torch.abs(inps.to(torch.float64))) )
        
            handles = []
            for name in subset:
                if name not in ['self_attn.out_proj','fc2']:
                    continue
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            
            for j in range(args.nsamples):
                outs[j] = (layer(inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask)[0]).to(dev2)
                outs2[j] = (layer(inps2[j].unsqueeze(0).to(dev), attention_mask=attention_mask)[0]).to(dev2)
       
            for h in handles:
                h.remove()
            
            for name in subset:
                if name not in ['self_attn.out_proj','fc2']:
                    continue
                osscar[name].get_XTY()

            inps2, outs2 = outs2, inps2
        
            # generate XTX
        
            handles = []
            for name in subset:
                if name not in ['self_attn.out_proj','fc2']:
                    continue
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            for j in range(args.nsamples):
                outs[j] = (layer(inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask)[0]).to(dev2)
                outs[j] = (layer(inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask)[0]).to(dev2)
            
            for h in handles:
                h.remove()
        else: 
            handles = []
            for name in subset:
                if name not in ['self_attn.out_proj','fc2']:
                    continue
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            for j in range(args.nsamples):
                outs[j] = (layer(inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask)[0]).to(dev2)
            
            for h in handles:
                h.remove()
        # start

        for name in subset:
            if name not in ['self_attn.out_proj','fc2']:
                continue
            print(i, name)
            osscar[name].prune(sp_fc=args.sp, sp_out=args.sp_out)
            osscar[name].free()
        
        if not parallel:
            for j in range(args.nsamples):
                outs[j] = (layer(inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask)[0]).to(dev2)
        
        layers[i] = layer.cpu()
        
        del layer
        del osscar 
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        
        
        

    model.config.use_cache = use_cache
 


    return 




@torch.no_grad()
def opt_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen
    
    print("number of nsamples is ",nsamples)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
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
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
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

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache
    return ppl.item()







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
        'sp', type=float, 
        help='Sparsity level'
    )

    parser.add_argument(
        '--sp_out',
        type=float, default=-1, help='Sparsity level for output layers'
    )

    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    
    parser.add_argument(
        '--algo',
        type=str, default="MP", help='Algorithms for pruning.'
    )


    parser.add_argument(
        '--model_path',
        type=str, default='./model', help='Path to the cached model.'
    )
    
    parser.add_argument(
        '--data_path',
        type=str, default='./data/', help='Path to the cached data.'
    )

    parser.add_argument(
        '--results_path',
        type=str, default='./results', help='Where to save results.'
    )

    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    
    parser.add_argument(
        '--testnsamples', type=int, default=256,
        help='Number of data samples for testing.'
    )
    
    parser.add_argument(
        '--parallel', action='store_true',
        help='Whether parallel or sequential.'
    )
    
    parser.add_argument(
        '--fullseq', action='store_true',
        help='Whether full sequential.'
    )
    
    parser.add_argument(
        '--local_out', action='store_true',
        help='Whether perform local search on output layers.'
    )
        
    parser.add_argument(
        '--local_fc', action='store_true',
        help='Whether perform local search on fc layers'
    )
    
    parser.add_argument(
        '--local_iter', type=int, default=30,
        help='number of iterations of local search'
    )
    
    parser.add_argument(
        '--local_test',type=int, default=5,
        help='number of tests of local search'
    )
    
    parser.add_argument(
        '--update_iter', type=int, default=10,
        help='Support update frequency.'
    )
    
    parser.add_argument(
        '--update_iter2', type=int, default=2,
        help='Support update frequency.'
    )
    
    parser.add_argument(
        '--lambda2', type=float, default=1e-2,
        help='Regular term'
    )


    

    args = parser.parse_args()
    
    if args.sp_out == -1:
        args.sp_out = args.sp
    
    head_list = {"facebook/opt-125m":12,"facebook/opt-350m":16,"facebook/opt-1.3b":32,"facebook/opt-2.7b":32,"facebook/opt-6.7b":32,
            "facebook/opt-13b":40,"facebook/opt-30b":56,"facebook/opt-66b":72}
    num_heads = head_list[args.model]

    model = get_opt(args.model, path = args.model_path) # Read the model
    model.eval() # Put to eval mode, no gradients are used.

    
    dataloader, testloader = get_loaders(
        args.dataset, data_path=args.data_path, nsamples=args.nsamples, testnsamples=args.testnsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    ) # Calibration data


    tick = time.time()
    opt_sequential(model, dataloader, DEV, num_heads, nsamples=args.nsamples, parallel=args.parallel)
    runtime = time.time() - tick

    tick = time.time()
    datasets = ['wikitext2', 'ptb', 'c4'] # Test datasets
    ppl = {}
    for dataset in datasets: 
        dataloader, testloader = get_loaders(
            dataset, data_path=args.data_path, seed=0, model=args.model, seqlen=model.seqlen, nsamples=args.nsamples, testnsamples=args.testnsamples,
        ) 
        ppl[dataset] = opt_eval(model, testloader, DEV) # Evaluate on test data
    infertime = time.time() - tick
    print("Results:",ppl)
    print("Run time:", runtime, "Inference time:",infertime)
    
    
    save_res = []
    save_res.append({
        'algo':args.algo,
        'sparsity':args.sp,
        'sparsity_out':args.sp_out,
        'local_out': args.local_out,
        'local_fc': args.local_fc,
        'model':args.model, 
        'seed':args.seed,
        'nsamples':args.nsamples,
        'lambda2':args.lambda2,
        'update_iter':args.update_iter,
        'update_iter2':args.update_iter2,
        'parallel':args.parallel,
        'wiki':ppl['wikitext2'],
        'ptb':ppl['ptb'],
        'c4':ppl['c4'],
        'time':runtime
    })
    
    FOLDER = args.results_path+'/' +'{}/{}_{}_{}_{}_{}_{}_{}_{}_{}/'.format((args.model).split("/")[1], args.algo, "parallel" if args.parallel else "sequential", args.sp, args.sp_out, args.update_iter, args.local_out, args.local_fc, args.lambda2,args.nsamples)
    os.makedirs(FOLDER, exist_ok=True)
    FILE = FOLDER+'data_{}.csv'.format(str(int(time.time())))  
    with open(FILE, "w") as file:
        json.dump(save_res, file)
        
    
    #filename_base = 'model_' + args.model.replace('/','--') + '_' + str(args.sp) + '_' + str(args.seed) + '_' + str(args.nsamples) + "_" + str(args.algo)  
    #filename = filename_base  +'.pt'
    #addr = "/home/gridsan/xmeng/Sparse_NN_shared/LLM/model_structured_xmeng_rebuttal/"+ filename
    #print(addr)
    #torch.save(model, addr)

