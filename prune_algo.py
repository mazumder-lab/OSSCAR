import numpy as np
import torch
import time


def MP_torch(W, XTX, XTY, num_cin, num_sp):
    
    totp, num_cout = W.shape
    ksize = int(totp / num_cin)
    
    W = W.reshape(num_cin, ksize, num_cout)
    idx = torch.argsort(torch.sum(torch.sum(torch.abs(W),axis=2),axis=1))
    W[idx[:num_cin-num_sp],:,:] = 0
    W = W.reshape(totp, num_cout)

    return W, torch.sum( -W * XTY + (1/2) * W * (XTX@W) )


def MP_plus_torch(W, XTX, XTY, num_cin, num_sp):
    
    totp, num_cout = W.shape
    ksize = int(totp / num_cin)
    
    W = W.reshape(num_cin, ksize, num_cout)
    idx = torch.argsort(torch.sum(torch.sum(torch.abs(W),axis=2),axis=1))
    W[idx[:num_cin-num_sp],:,:] = 0
    W = W.reshape(totp, num_cout)

    W_sol = torch.zeros_like(W)
    nzi = torch.nonzero(W[:,0], as_tuple=True)[0]
    XTX_sub = XTX[nzi[:,None],nzi]
    XTY_sub = XTY[nzi,:]
    
    W_sol[nzi,:] = torch.linalg.inv(XTX_sub)@ XTY_sub
    return W_sol, torch.sum( -W_sol * XTY + (1/2) * W_sol * (XTX@W_sol) )




def OSSCAR_fastprune(W, XTX, XTY, num_cin, num_sp, update_iter = 1):
    
    DEV = W.device
    Wtype = W.dtype
    totp, num_cout = W.shape
    ksize = int(totp / num_cin)
    
    W = W.to(torch.float64)
    XTX = XTX.to(torch.float64)
    XTY = XTY.to(torch.float64)
    
    XTX_inv = torch.linalg.inv(XTX)
    
    
    
    num_prune = torch.sum(torch.abs(torch.sum(torch.sum(W.reshape(num_cin, ksize, num_cout),axis=2),axis=1)) <= 1e-12)
    prune_list = torch.abs(torch.sum(torch.sum(W.reshape(num_cin, ksize, num_cout),axis=2),axis=1)) <= 1e-12
    
    if num_prune:
        upd_idx = torch.cat([torch.arange(i*ksize, (i+1)*ksize) for i in range(num_cin) if prune_list[i]])
        XTX_inv[upd_idx,:] = 0
        XTX_inv[:,upd_idx] = 0
    
    #W = XTX_inv@ XTY
    
    if int(num_cin-num_sp-num_prune) <= 0:
        upd_it = 0
    else:
        upd_it = int((num_cin-num_sp-num_prune) / update_iter)
        if upd_it == 0:
            upd_it = 1
        quo, rem = divmod(int(num_cin-num_sp-num_prune), int(upd_it))
        update_ten = torch.full((upd_it,), quo, dtype=torch.int).to(DEV)
        update_ten[:rem] += 1
    
    
    for i1 in range(upd_it):
        
        obj_mat = torch.zeros_like(W)
        if ksize > 1:
            for i2 in range(num_cin):
                if prune_list[i2]:
                    continue
                obj_mat[i2*ksize:(i2+1)*ksize,:] = torch.linalg.inv(XTX_inv[i2*ksize:(i2+1)*ksize,i2*ksize:(i2+1)*ksize])@W[i2*ksize:(i2+1)*ksize,:] / 2
        else:
            obj_mat = (1 / (prune_list + torch.diag(XTX_inv)))[:,None] * W / 2
            
        obj_cha = W * obj_mat
        obj_cha = obj_cha.reshape(num_cin, ksize, num_cout)
        obj_sum = torch.sum(torch.sum(obj_cha,axis=2),axis=1)
        
        idx = torch.argsort(obj_sum + 1e20 * (prune_list) )
        
        

        upd_idx = torch.cat([torch.arange(idx[i]*ksize, (idx[i]+1)*ksize) for i in range(update_ten[i1])])

        
        Xinv_tmp = torch.linalg.inv(XTX_inv[upd_idx[:,None],upd_idx])
        
        W -= XTX_inv[:,upd_idx] @ Xinv_tmp @ W[upd_idx,:]
        W = W.reshape(num_cin, ksize, num_cout)
        W[idx[:update_ten[i1]],:,:] = 0
        W = W.reshape(totp, num_cout)
    
        XTX_inv -= XTX_inv[:,upd_idx] @ Xinv_tmp @ XTX_inv[upd_idx,:]
        XTX_inv[upd_idx,:] = 0
        XTX_inv[:,upd_idx] = 0
    
        prune_list[idx[:update_ten[i1]]] = True
        
         
    W_sol = torch.zeros_like(W)
    nzi = torch.nonzero(W[:,0], as_tuple=True)[0]
    W_sol[nzi,:] = torch.linalg.inv(XTX[nzi[:,None],nzi])@ XTY[nzi,:]
    
    W_sol = W_sol.to(Wtype)
    XTY = XTY.to(Wtype)
    XTX = XTX.to(Wtype)
    
    return W_sol, torch.sum( -W_sol * XTY + (1/2) * W_sol * (XTX@W_sol) ) 



def OSSCAR_local_search(W, XTX, XTY, num_cin, max_iter = 20, num_swap = 100, switch_lb = 1):
    
    DEV = W.device
    Wtype = W.dtype
    totp, num_cout = W.shape
    
    W = W.to(torch.float64)
    XTX = XTX.to(torch.float64)
    XTY = XTY.to(torch.float64)
    
    #num_swap = int(np.ceil(num_cin * switch_ratio))
    lb_swap = int(np.ceil(num_cin * switch_lb))
    
    ksize = int(totp / num_cin)
    
    prune_list = torch.abs(torch.sum(torch.sum(W.reshape(num_cin, ksize, num_cout),axis=2),axis=1)) <= 1e-12
       
    best_prune = torch.clone(prune_list)
    supp_idx = torch.cat([torch.arange(i*ksize, (i+1)*ksize) for i in range(num_cin) if not prune_list[i]])
    
    XTX_inv = torch.zeros_like(XTX)
    XTX_inv[supp_idx[:,None],supp_idx] = torch.linalg.inv(XTX[supp_idx[:,None],supp_idx])

    
    obj_cur = torch.sum( -W * XTY + (1/2) * W* (XTX@W) ) 
    
    for i_local in range(max_iter):

        obj_mat = torch.zeros_like(W)
        if ksize > 1:
            for i2 in range(num_cin):
                if prune_list[i2]:
                    continue
                obj_mat[i2*ksize:(i2+1)*ksize,:] = torch.linalg.inv(XTX_inv[i2*ksize:(i2+1)*ksize,i2*ksize:(i2+1)*ksize])@W[i2*ksize:(i2+1)*ksize,:] / 2
        else:
            obj_mat = (1 / (prune_list + torch.diag(XTX_inv)))[:,None] * W / 2


        obj_cha = W * obj_mat
        obj_cha = obj_cha.reshape(num_cin, ksize, num_cout)
        obj_sum = torch.sum(torch.sum(obj_cha,axis=2),axis=1)
        
        idx = torch.argsort(obj_sum + 1e20 * (prune_list) )
        
        upd_idx = torch.cat([torch.arange(idx[i]*ksize, (idx[i]+1)*ksize) for i in range(num_swap)])

        
        Xinv_tmp = torch.linalg.inv(XTX_inv[upd_idx[:,None],upd_idx])
        W -= XTX_inv[:,upd_idx] @ Xinv_tmp @ W[upd_idx,:]
        W = W.reshape(num_cin, ksize, num_cout)
        W[idx[:num_swap],:,:] = 0
        W = W.reshape(totp, num_cout)
    
        XTX_inv -= XTX_inv[:,upd_idx] @ Xinv_tmp @ XTX_inv[upd_idx,:]
        XTX_inv[upd_idx,:] = 0
        XTX_inv[:,upd_idx] = 0
    
        prune_list[idx[:num_swap]] = True
        
        
            
        obj_in = torch.zeros((num_cin,),dtype=torch.float64).to(DEV)

            
        supp_idx = torch.cat([torch.arange(i*ksize, (i+1)*ksize) for i in range(num_cin) if not prune_list[i]])
        H_inv = XTX_inv[supp_idx[:,None],supp_idx]
        H_invG = H_inv @ XTY[supp_idx,:]
            
            

        if ksize >= 2:
            for i3 in range(num_cin):
                if not prune_list[i3]:
                    continue
                        
                    
                b_ori = XTX[supp_idx,i3*ksize:(i3+1)*ksize]
                C_inv = torch.linalg.inv(XTX[i3*ksize:(i3+1)*ksize,i3*ksize:(i3+1)*ksize] - b_ori.T @ H_inv @ b_ori)
                    
                gt = XTY[i3*ksize:(i3+1)*ksize,:] - b_ori.T @ H_invG
                obj_in[i3] = torch.sum(gt * (C_inv @ gt)) / 2
                    
                #W1 = torch.clone(W)
                #W1[i2*ksize:(i2+1)*ksize,:] = 0
                #nzi = torch.nonzero(W1[:,0], as_tuple=True)[0]
                #XTX_sub = XTX[nzi[:,None],nzi]
                #XTY_sub = XTY[nzi,:]
                #W1[nzi,:] = torch.linalg.inv(XTX_sub)@ XTY_sub
                #obj1 = torch.sum( -W1 * XTY + (1/2) * W1 * (XTX @ W1)) 
                    
                #W2 = torch.clone(W)
                #W2[i2*ksize:(i2+1)*ksize,:] = 0
                #W2[i3*ksize:(i3+1)*ksize,:] = 1
                #nzi = torch.nonzero(W2[:,0], as_tuple=True)[0]
                #XTX_sub = XTX[nzi[:,None],nzi]
                #XTY_sub = XTY[nzi,:]
                #W2[nzi,:] = torch.linalg.inv(XTX_sub)@ XTY_sub
                #obj2 = torch.sum( -W2 * XTY + (1/2) * W2 * (XTX @ W2)) 
                    
                #print("Out: {}, in: {}, true obj is ori: {}, out: {}".format(i2,i3,obj1-obj_cur,obj1-obj2))
                #print("Out: {}, in: {}, estimate obj is out: {}, in: {}".format(i2,i3,obj_sum[i2],obj_in[i3]))
                         
 
        else:
            C_list = 1 / (torch.diag(XTX) -  torch.sum(XTX[supp_idx,:] * (H_inv @ XTX[supp_idx,:] ), axis = 0) + (~prune_list)*1e-8 ) 
            gt = XTY - XTX[:,supp_idx] @ H_invG
            obj_in = torch.sum(gt**2, axis=1) * C_list / 2
            
        
        idx2 = torch.argsort(-obj_in + 1e20*(~prune_list))
        
        prune_list[idx2[:num_swap]] = False   
        supp_idx = torch.cat([torch.arange(i*ksize, (i+1)*ksize) for i in range(num_cin) if not prune_list[i]])
    
        XTX_inv = torch.zeros_like(XTX)
        XTX_inv[supp_idx[:,None],supp_idx] = torch.linalg.inv(XTX[supp_idx[:,None],supp_idx])
                
        W = torch.zeros_like(W)
        W = XTX_inv @ XTY
                
        obj_new = torch.sum( -W * XTY + (1/2) * W * (XTX @ W))         
                
        print("Finish iter {}, old obj is {}, new is {}, numswap is {}".format(i_local, obj_cur, obj_new, num_swap))
        if obj_new < obj_cur * (1+1e-9):
            
            best_prune = torch.clone(prune_list)
            obj_cur = obj_new
        else:
            if switch_lb >=1 or num_swap <= lb_swap:
                break
            else:
                num_swap = int(np.maximum(num_swap/2, lb_swap))
                
                prune_list = torch.clone(best_prune)
                supp_idx = torch.cat([torch.arange(i*ksize, (i+1)*ksize) for i in range(num_cin) if not prune_list[i]])
    
                XTX_inv = torch.zeros_like(XTX)
                XTX_inv[supp_idx[:,None],supp_idx] = torch.linalg.inv(XTX[supp_idx[:,None],supp_idx])
                
                W = torch.zeros_like(W)
                W = XTX_inv @ XTY

    
    supp_idx = torch.cat([torch.arange(i*ksize, (i+1)*ksize) for i in range(num_cin) if not best_prune[i]])
    
    W = torch.zeros_like(W)
    W[supp_idx,:] = torch.linalg.inv(XTX[supp_idx[:,None],supp_idx]) @ XTY[supp_idx,:]
    
    W = W.to(Wtype)
    XTY = XTY.to(Wtype)
    XTX = XTX.to(Wtype)
    
    return W, torch.sum( -W * XTY + (1/2) * W * (XTX@W) ) 

