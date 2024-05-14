import math
import time

import math
import torch
import torch.nn as nn
import transformers
import numpy

from prune_algo import *


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class OSSCAR_prune:

    def __init__(self, layer, algo, nsamples, seqlen, update_iter=1, update_iter2=1, lambda2=1e-2, layername=None, num_heads = 32, local_out = False, local_fc = False, local_iter = 20, local_test = 10, fullseq=False):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.algo = algo
        self.update_iter = update_iter
        self.update_iter2 = update_iter2
        self.lambda2 = lambda2
        self.nsamples = nsamples
        self.seqlen = seqlen
        self.layername = layername
        self.num_heads = num_heads
        self.equi_nsamples = self.nsamples*self.seqlen
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.local_out = local_out
        self.local_fc = local_fc
        self.local_iter = local_iter
        self.local_test = local_test
        self.fullseq = fullseq

        self.XtX = torch.zeros((self.columns, self.columns), device=self.dev)

        self.count = 0

        self.nsamples0 = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

            if len(out.shape) == 3:
                out = out.reshape((-1, out.shape[-1]))
        out = out.t()
        if isinstance(self.layer, nn.Conv2d):
            print(inp.shape)
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        inp = inp.float()
        out = out.float()

        self.nsamples0 += tmp
        if self.fullseq:
            if self.nsamples0 % 2 == 0:
                self.XtX += (self.inp).matmul(inp.t()) / tmp
                self.inp = None
            else:
                self.inp = inp
        else:
            self.XtX += (inp).matmul(inp.t()) / tmp
        
        
        
        
    def get_XTY(self):
        
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        
        dead = torch.diag(self.XtX) == 0
        B = W.t()
        B[dead,:] = 0
        
        
        self.XtX += torch.eye(B.shape[0]).to("cuda") * self.lambda2 * torch.mean(torch.diag(self.XtX))
        
        if "torch" not in self.algo:
            self.XtY = (self.XtX @ B).cpu().numpy()
        else:
            self.XtY = self.XtX @ B
            
            
        
        
        if isinstance(self.layer, transformers.Conv1D):
            self.layer.weight.data = B.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        else:
            self.layer.weight.data = B.t().reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        self.XtX = torch.zeros((self.columns, self.columns), device=self.dev)
        print("number of samples: ",self.nsamples0)
        
        self.nsamples0 = 0
        W = None
        B = None


        
        
    def prune(self, sp_fc, sp_out):

        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        
        print("Layer:",self.layer, flush=True)
       
        st_time = time.time()

        

        dead = torch.diag(self.XtX) == 0
        B = W.t()
        B[dead,:] = 0
           
            
        self.XtX += torch.eye(B.shape[0]).to("cuda") * self.lambda2 * torch.mean(torch.diag(self.XtX))
            
        if not self.fullseq:
            self.XtY = (self.XtX @ B)
        print("num of dead is", torch.sum(dead))
            
            
            
        pre_time = time.time() - st_time
        st_time = time.time()
        
        if self.layername != "self_attn.out_proj":
            num_cin = B.shape[0]
            sp = sp_fc
            upd_iter = self.update_iter
            local_swap = self.update_iter
            if self.local_fc:
                use_local = True
            else:
                use_local = False
        else:
            num_cin = self.num_heads
            sp = sp_out
            upd_iter = self.update_iter2
            local_swap = 5
            if self.local_out:
                use_local = True
            else:
                use_local = False
            
        print("sp is {}, num_cin is {}".format(sp,num_cin))


        if self.algo == "MP":
            
            B_sol, B_obj = MP_torch(B, self.XtX, self.XtY, num_cin, int(num_cin * (1-sp)))
            
        elif self.algo == "MP_plus":
            
            B_sol, B_obj = MP_plus_torch(B, self.XtX, self.XtY, num_cin, int(num_cin * (1-sp)))
                
        elif self.algo == "OSSCAR_prune":
            
            B_sol, B_obj = OSSCAR_fastprune(B.clone(), self.XtX, self.XtY, num_cin, int(num_cin * (1-sp)), upd_iter)
            if use_local:
                B_sol, B_obj = OSSCAR_local_search(B_sol, self.XtX, self.XtY, num_cin, self.local_iter, local_swap)
                

        
        run_time = time.time() - st_time
        
        if "torch" not in self.algo:
            B = torch.Tensor(B_sol).to("cuda")
        else:
            B = B_sol
            
        print("num of zeros:", torch.sum(B==0))
            
        
        print("pre-processing time: ",pre_time, "alg time: ",run_time)
        self.pre_time = pre_time
        self.run_time = run_time
        
        if isinstance(self.layer, transformers.Conv1D):
            self.layer.weight.data = B.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        else:
            self.layer.weight.data = B.t().reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
            

        return 



    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        self.X = None
        self.Y = None
        self.XtX = None
        self.YXt = None
        self.YtX = None
        self.XtY = None
        torch.cuda.empty_cache()
