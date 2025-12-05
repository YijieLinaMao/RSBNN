import os
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split
import torch.optim as optim
import time
import datetime
import json
import copy
os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class normalized_P_SDMA(nn.Module):
    def __init__(self,bs_antenna_num,user_num,power_constr):
        super(normalized_P_SDMA, self).__init__()
        self.bs_antenna_num=bs_antenna_num
        self.user_num=user_num
        self.power_constr=power_constr
    def recombine_P(self,P,batch_size):
        P_new = torch.zeros((self.bs_antenna_num,self.user_num,batch_size))+1j*torch.zeros((self.bs_antenna_num,self.user_num,batch_size))
        for sample_index in range(batch_size):
            real = (P[sample_index ,:(self.user_num) * self.bs_antenna_num]).reshape(self.bs_antenna_num, self.user_num)
            imag = (P[sample_index ,(self.user_num) * self.bs_antenna_num:]).reshape(self.bs_antenna_num, self.user_num)
            P_new[:, :, sample_index] = real + 1j * imag
        return P_new
    def normalize_P(self, predict_beam_dl, batch_size):
        batch_size=predict_beam_dl.shape[0]
        P_new = torch.zeros_like(predict_beam_dl)
        for sample_index in range(batch_size):
            temp = predict_beam_dl[sample_index,:, :]
            temp2 = self.power_constr / torch.trace(temp @ torch.conj(temp.T))
            P_new[sample_index,:, :] = torch.sqrt(temp2) * temp
        return P_new
    def recover_P(self,predict_beam_dl):
        sample_num=predict_beam_dl.shape[-1]
        sample_beam = torch.zeros((sample_num, self.user_num * self.bs_antenna_num * 2))
        for sample_index in range(sample_num):
            sample_beam[sample_index, :self.user_num * self.bs_antenna_num] = predict_beam_dl[:, :, sample_index].reshape(-1,self.user_num * self.bs_antenna_num).real.float()
            sample_beam[sample_index, self.user_num * self.bs_antenna_num:] = predict_beam_dl[:, :, sample_index].reshape(-1,self.user_num * self.bs_antenna_num).imag.float()
        return sample_beam
    def forward(self,P):
        batch_size=P.shape[0]
        P_new=self.recombine_P(P,batch_size)
        predict_beam_dl=self.normalize_P(P_new,batch_size)
        P_recover=self.recover_P(predict_beam_dl)
        return P_recover

class normalized_P_1RS(normalized_P_SDMA):
    def __init__(self,bs_antenna_num,user_num,power_constr):
        super(normalized_P_SDMA, self).__init__()
        self.bs_antenna_num=bs_antenna_num
        self.user_num=user_num
        self.power_constr=power_constr
    def recombine_P(self,P,batch_size):
        P_new = torch.zeros((self.bs_antenna_num,self.user_num+1,batch_size))+1j*torch.zeros((self.bs_antenna_num,self.user_num+1,batch_size))
        for sample_index in range(batch_size):
            real = (P[sample_index ,:(self.user_num+1) * self.bs_antenna_num]).reshape(self.bs_antenna_num, self.user_num+1)
            imag = (P[sample_index ,(self.user_num+1) * self.bs_antenna_num:]).reshape(self.bs_antenna_num, self.user_num+1)
            P_new[:, :, sample_index] = real + 1j * imag
        return P_new
    def recover_P(self,predict_beam_dl):
        sample_num=predict_beam_dl.shape[-1]
        sample_beam = torch.zeros((sample_num, (user_num+1) * bs_antenna_num * 2))
        for sample_index in range(sample_num):
            sample_beam[sample_index, :(user_num+1) * bs_antenna_num] = predict_beam_dl[:, :, sample_index].reshape(-1,(user_num+1) * bs_antenna_num).real.float()
            sample_beam[sample_index, (user_num+1) * bs_antenna_num:] = predict_beam_dl[:, :, sample_index].reshape(-1,(user_num+1) * bs_antenna_num).imag.float()
        return sample_beam

class loss_DBL_SR_SDMA(nn.Module):
    def __init__(self,bs_antenna_num,user_num,power_constr,noise_variance):
        super(loss_DBL_SR_SDMA, self).__init__()
        self.bs_antenna_num=bs_antenna_num
        self.user_num=user_num
        self.power_constr=power_constr
        self.noise_variance=noise_variance
    def transform_H(self,H):
        self.batchsize=H.size(dim=0)
        H_new=torch.complex(torch.zeros(self.bs_antenna_num,self.user_num,H.size(dim=0)),torch.zeros(self.bs_antenna_num,self.user_num,H.size(dim=0)))
        for i in range(self.batchsize):
            h=H[i,:,:,:]
            h= torch.complex(h[:,:,0], h[:,:,1])
            h=torch.reshape(h, (self.bs_antenna_num,self.user_num))
            H_new[:,:,i]=h
        return H_new
    def normalize_P(self,predict_beam_dl,batch_size):
        P_new=torch.zeros_like(predict_beam_dl)
        for sample_index in range(batch_size):
            temp=predict_beam_dl[:,: , sample_index]
            temp2 = self.power_constr / torch.trace(temp @ torch.conj(temp.T))
            P_new[:, :, sample_index]=torch.sqrt(temp2)*temp
        return P_new
    def recombine_P(self,P,batch_size):
        P_new = torch.zeros((self.bs_antenna_num,self.user_num,batch_size))+1j*torch.zeros((self.bs_antenna_num,self.user_num,batch_size))
        for sample_index in range(batch_size):
            real = (P[sample_index ,:self.user_num * self.bs_antenna_num]).reshape(self.bs_antenna_num, self.user_num)
            imag = (P[sample_index ,self.user_num * self.bs_antenna_num:]).reshape(self.bs_antenna_num, self.user_num)
            P_new[:, :, sample_index] = real + 1j * imag
        return P_new
    def transform_P(self,batch_size,P,H):
        P_new=self.recombine_P(P,batch_size)
        predict_beam_dl=P_new
        return predict_beam_dl
    def forward(self, H,P):
        batch_size=H.size(dim=0)
        H=self.transform_H(H)
        predict_sumrate = torch.zeros((1, batch_size))
        predict_beam_dl=self.transform_P(batch_size,P,H)
        for sample_index in range(batch_size):
            for user_index in range(self.user_num):
                sig = torch.square(torch.norm(predict_beam_dl[:, user_index, sample_index].matmul(
                    H[:,user_index,sample_index])))
                interf = - sig + torch.square(torch.norm(
                    H[:, user_index, sample_index].matmul(predict_beam_dl[:, :, sample_index])))+self.noise_variance
                predict_sumrate[0, sample_index] = predict_sumrate[0, sample_index] + torch.log2(1 + sig / interf)
                if not torch.isnan(predict_sumrate).any() or not torch.isinf(predict_sumrate[0,sample_index]):
                    pass
        return -torch.sum(predict_sumrate)/batch_size

class loss_BNN_SR_SDMA(loss_DBL_SR_SDMA):
    def __init__(self,bs_antenna_num,user_num,power_constr,noise_variance):
        super(loss_BNN_SR_SDMA, self).__init__(bs_antenna_num,user_num,power_constr,noise_variance)
    def transform_P(self,batch_size,P,H):
        predict_dl_power, predict_ul_power=self.recombine_P(P,batch_size)
        predict_beam_dl=self.reconstruct_P(batch_size,predict_ul_power,predict_dl_power,H)
        return predict_beam_dl

    def recombine_P(self,P,batch_size):
        sample_num=P.shape[0]
        predict_ul_power = torch.zeros((self.user_num, sample_num))
        predict_dl_power = torch.zeros((self.user_num, sample_num))
        for sample_index in range(batch_size):
            factor = self.power_constr / torch.norm(P[sample_index, 0:user_num], 1)
            predict_ul_power[:, sample_index] = factor * P[sample_index, 0:user_num].T
            factor = self.power_constr / torch.norm(P[sample_index, user_num:2 * user_num], 1)
            predict_dl_power[:, sample_index] = factor * P[sample_index, user_num:2 * user_num].T
        return predict_dl_power,predict_ul_power
    def reconstruct_P(self,batch_size,predict_ul_power,predict_dl_power,H):
        predict_beam_dl = torch.complex(torch.zeros((bs_antenna_num, user_num, batch_size)),torch.zeros((bs_antenna_num, user_num, batch_size)))
        for sample_index in range(batch_size):
            TT = self.noise_variance * torch.eye(bs_antenna_num)
            for user_index in range(user_num):
                TT = TT + predict_ul_power[user_index, sample_index] * (
                    torch.mm(torch.conj(H[:, user_index, sample_index].reshape(-1, 1)),
                           H[:, user_index, sample_index].reshape(1, -1)))
            for user_index in range(user_num):
                beam_up_temp = torch.matmul(torch.inverse(TT),
                                      torch.conj(H[:, user_index, sample_index]))
                predict_beam = beam_up_temp / torch.norm(beam_up_temp)
                temp = torch.sqrt(predict_dl_power[user_index, sample_index]) * predict_beam
                predict_beam_dl[:, user_index, sample_index] = temp
        return predict_beam_dl

class loss_DBL_SR_1RS(loss_DBL_SR_SDMA):
    def __init__(self,bs_antenna_num,user_num,power_constr,noise_variance):
        super(loss_DBL_SR_1RS, self).__init__(bs_antenna_num,user_num,power_constr,noise_variance)
    def transform_P(self, batch_size, P,H):
        P_new = self.recombine_P(P, batch_size)
        predict_beam_dl = self.normalize_P(P_new, batch_size)
        return predict_beam_dl
    def forward(self, H,P):
        batch_size=H.size(dim=0)
        H=self.transform_H(H)
        predict_sumrate = torch.zeros((1, batch_size))
        predict_beam_dl=self.transform_P(batch_size,P,H)
        P_c,P_p=predict_beam_dl[:,0,:][:,np.newaxis],predict_beam_dl[:,1:,:]
        for sample_index in range(batch_size):
            com_rate=torch.zeros(self.user_num)
            for user_index in range(self.user_num):
                sig = torch.square(torch.norm(P_p[:, user_index, sample_index].matmul(
                    H[:,user_index,sample_index])))
                interf = - sig + torch.square(torch.norm(
                    H[:, user_index, sample_index].matmul(P_p[:, :, sample_index])))+self.noise_variance
                com_interf=sig+interf
                com_sig=torch.square(torch.norm(P_c[:, 0, sample_index].matmul(
                    H[:,user_index,sample_index])))
                com_rate[user_index]=torch.log2(1+com_sig/com_interf)
                predict_sumrate[0, sample_index] = predict_sumrate[0, sample_index] + torch.log2(1 + sig / interf)
            predict_sumrate[0, sample_index]+=torch.min(com_rate)
        return -torch.sum(predict_sumrate)/batch_size
    def recombine_P(self,P,batch_size):
        P_new = torch.zeros((self.bs_antenna_num,self.user_num+1,batch_size))+1j*torch.zeros((self.bs_antenna_num,self.user_num+1,batch_size))
        for sample_index in range(batch_size):
            real = (P[sample_index ,:(self.user_num+1) * self.bs_antenna_num]).reshape(self.bs_antenna_num, self.user_num+1)
            imag = (P[sample_index ,(self.user_num+1) * self.bs_antenna_num:]).reshape(self.bs_antenna_num, self.user_num+1)
            P_new[:, :, sample_index] = real + 1j * imag
        return P_new

class loss_BNN_SR_1RS(loss_DBL_SR_1RS):
    def __init__(self,bs_antenna_num,user_num,power_constr,noise_variance):
        super(loss_BNN_SR_1RS, self).__init__(bs_antenna_num,user_num,power_constr,noise_variance)
    def recombine_P(self,P,batch_size):
        sample_num=P.shape[0]
        predict_ul_power = torch.zeros((self.user_num, sample_num))
        predict_dl_power = torch.zeros((self.user_num, sample_num))
        predict_ul_power_c = torch.zeros((self.user_num, sample_num))
        predict_dl_power_c = torch.zeros((self.user_num, sample_num))

        for sample_index in range(batch_size):
            predict_ul_power[:, sample_index] =  P[sample_index, 0:self.user_num].T
            predict_dl_power[:, sample_index] =  P[sample_index, self.user_num:2 * self.user_num].T
            predict_ul_power_c[:, sample_index] =  P[sample_index, 2*self.user_num:3 * self.user_num].T
            predict_dl_power_c[:, sample_index] =  P[sample_index, 3*self.user_num:4 * self.user_num].T

        return predict_dl_power,predict_ul_power,predict_dl_power_c,predict_ul_power_c
    def reconstruct_P(self,batch_size,a,b,H,c=None,d=None):
        predict_beam_dl = torch.complex(torch.zeros((bs_antenna_num, user_num+1, batch_size)),torch.zeros((bs_antenna_num, user_num+1, batch_size)))
        for sample_index in range(batch_size):
            TT = self.noise_variance * torch.eye(bs_antenna_num)
            TT2=self.noise_variance * torch.eye(bs_antenna_num)
            for user_index in range(user_num):
                H_square=(torch.mm(torch.conj(H[:, user_index, sample_index].reshape(-1, 1)),
                           H[:, user_index, sample_index].reshape(1, -1)))
                TT = TT + (a[user_index, sample_index]+b[user_index, sample_index]) * H_square
                TT2=TT2+(b[user_index, sample_index])* H_square
            temp_c=torch.complex(torch.zeros((bs_antenna_num, )),torch.zeros((bs_antenna_num, )))
            for user_index in range(user_num):
                beam_up_temp = torch.matmul(torch.inverse(TT),
                                      torch.conj(H[:, user_index, sample_index]))
                temp = torch.sqrt(c[user_index, sample_index]) * beam_up_temp
                temp_c += torch.sqrt(d[user_index, sample_index]) * torch.conj(H[:, user_index, sample_index])
                predict_beam_dl[:, user_index+1, sample_index] = temp
            predict_beam_dl[:, 0, sample_index]=torch.matmul(torch.inverse(TT2),temp_c)
        return predict_beam_dl
    def transform_P(self,batch_size,P,H):
        a, b,c,d=self.recombine_P(P,batch_size)
        predict_beam_dl=self.reconstruct_P(batch_size,a,b,H,c,d)
        predict_beam_dl=self.normalize_P(predict_beam_dl,batch_size)
        return predict_beam_dl

class loss_unfold_SR_1RS(loss_DBL_SR_1RS):
    def __init__(self,bs_antenna_num,user_num,power_constr,noise_variance):
        super(loss_unfold_SR_1RS, self).__init__(bs_antenna_num,user_num,power_constr,noise_variance)
    def forward(self, H,P):
        batch_size=H.size(dim=0)
        H=torch.permute(H,(1,2,0))
        P=torch.permute(P,(1,2,0))
        predict_sumrate = torch.zeros((1, batch_size))
        P_c,P_p=P[:,-1,:][:,np.newaxis],P[:,0:user_num,:]
        for sample_index in range(batch_size):
            com_rate=torch.zeros(self.user_num)
            for user_index in range(self.user_num):
                sig = torch.square(torch.norm(P_p[:, user_index, sample_index].matmul(
                    torch.conj(H[:,user_index,sample_index]))))
                interf = - sig + torch.square(torch.norm(
                    torch.conj(H[:, user_index, sample_index]).matmul(P_p[:, :, sample_index])))+self.noise_variance
                com_interf=sig+interf
                com_sig=torch.square(torch.norm(P_c[:, 0, sample_index].matmul(
                    torch.conj(H[:,user_index,sample_index]))))
                com_rate[user_index]=torch.log2(1+com_sig/com_interf)
                predict_sumrate[0, sample_index] = predict_sumrate[0, sample_index] + torch.log2(1 + sig / interf)
                #if not torch.isnan(predict_sumrate).any() or not torch.isinf(predict_sumrate[0,sample_index]):
                    #pass
            predict_sumrate[0, sample_index]+=torch.min(com_rate)
        return -torch.sum(predict_sumrate)/batch_size

class loss_unfold_SR_1RS_supervised(nn.Module):
    def __init__(self):
        super(loss_unfold_SR_1RS_supervised, self).__init__()
        self.weight=[0.9,0.1]
    def forward(self,L,label):
        return torch.sum(torch.square(L-label))/L.shape[0]

class unfold_layer(nn.Module):
    def __init__(self,bs_antenna_num,user_num,power_constr,noise_variance,batch_size):
        super(unfold_layer, self).__init__()
        self.bs_antenna_num=bs_antenna_num
        self.user_num=user_num
        self.power_constr=power_constr
        self.noise_variance=noise_variance
        self.noise=torch.ones((user_num, 1))
        self.batch_size=batch_size
        self.net=base_net(user_num).to(device)
    def separate_P(self,predict_beam_dl):
        sample_num=predict_beam_dl.shape[0]
        sample_beam = torch.zeros(((user_num+1) * bs_antenna_num * 2,sample_num),device=device)
        for sample_index in range(sample_num):
            sample_beam[ :(user_num+1) * bs_antenna_num,sample_index] = predict_beam_dl[ sample_index,:, :].reshape(-1,(user_num+1) * bs_antenna_num).real.float()
            sample_beam[ (user_num+1) * bs_antenna_num:,sample_index] = predict_beam_dl[ sample_index,:, :].reshape(-1,(user_num+1) * bs_antenna_num).imag.float()
        return sample_beam
    def separate_H(self,H):
        sample_num=H.shape[0]
        sample_H= torch.zeros(((user_num) * bs_antenna_num * 2,sample_num),device=device)
        for sample_index in range(sample_num):
            sample_H[ :(user_num) * bs_antenna_num,sample_index] = H[ sample_index,:, :].reshape(-1,(user_num) * bs_antenna_num).real.float()
            sample_H[ (user_num) * bs_antenna_num:,sample_index] = H[ sample_index,:, :].reshape(-1,(user_num) * bs_antenna_num).imag.float()
        return sample_H
    def forward(self,precoders,HH,parameters_org):
        P_batch,p_c_batch=precoders[:,:,0:self.user_num],precoders[:,:,-1].unsqueeze(dim=2)
        noise=torch.ones((HH.shape[0],self.user_num,1),device=device)
        H_H = torch.conj(torch.transpose(HH,1,2))
        HP = torch.matmul(H_H, P_batch)
        T_k = torch.sum(torch.square(torch.abs(HP)), dim=2, keepdim=True) + noise
        Hpc = torch.matmul(H_H, p_c_batch)
        square_abs_Hp_c = torch.square(torch.abs(Hpc))
        T_ck = T_k + square_abs_Hp_c
        diag_HP = torch.diagonal(HP, offset=0, dim1=1, dim2=2).unsqueeze(2)
        abs_square = (torch.abs(diag_HP ** 2))
        alpha_k = T_k / (T_k - abs_square) - 1
        alpha_ck = torch.abs(Hpc) ** 2 / T_k
        beta_k = torch.sqrt(1 + alpha_k) * diag_HP / T_k
        beta_ck = torch.sqrt(1 + alpha_ck) * Hpc / T_ck
        P_separate=self.separate_P(precoders)
        H_separate=self.separate_H(HH)
        input_vec=torch.cat((torch.transpose(P_separate,dim0=0,dim1=1),torch.transpose(H_separate,dim0=0,dim1=1),parameters_org),dim=1)
        parameters_batch = self.net(input_vec)
        temp=parameters_batch[:,0:self.user_num]+0.01*torch.ones_like(parameters_batch[:,0:self.user_num])
        lam_batch=temp/torch.sum(temp,dim=1).unsqueeze(dim=1)
        lam_batch=lam_batch.unsqueeze(dim=2)
        mu_batch=parameters_batch[:,-1].unsqueeze(dim=1)
        trans_betak = torch.conj(beta_k.transpose(1, 2))
        trans_betack = beta_ck.transpose(1, 2)
        beta_k_square_abs = torch.abs(trans_betak) ** 2
        beta_ck_square_abs = torch.abs(torch.conj(trans_betack)) ** 2
        lambda_i_H = torch.conj(lam_batch.transpose(1, 2))
        H_k = torch.sqrt(beta_k_square_abs + lambda_i_H * beta_ck_square_abs) * HH
        H_ck = torch.sqrt(lambda_i_H * beta_ck_square_abs) * HH
        T_Hk = torch.conj(H_k.transpose(1, 2))
        T_Hck = torch.conj(H_ck.transpose(1, 2))
        p_c_one = torch.matmul(H_ck, T_Hck)+ mu_batch[:, None] * torch.eye(self.bs_antenna_num, device=device)[None, :, :]
        t_alphack = torch.conj(alpha_ck.transpose(1, 2))
        p_c_two = lambda_i_H * torch.sqrt(1 + t_alphack) * trans_betack * HH
        p_c_two = torch.sum(p_c_two, dim=2, keepdim=True)
        P_one = torch.matmul(H_k, T_Hk) + mu_batch[:, None] * torch.eye(self.bs_antenna_num, device=device)[None, :, :]
        P_two = torch.transpose(torch.sqrt(1 + alpha_k) * beta_k, 1, 2) * HH
        p_c_batch = torch.matmul(torch.inverse(p_c_one), p_c_two)
        P_batch = torch.matmul(torch.inverse(P_one), P_two)
        new_precoders=torch.concatenate((P_batch,p_c_batch),dim=2)
        new_parameter=torch.zeros_like(parameters_batch)
        new_parameter[:,0:self.user_num]=lam_batch.squeeze(dim=2)
        new_parameter[:, -1]=mu_batch.squeeze(dim=1)
        return new_precoders,new_parameter

class unfold_net(nn.Module):
    def __init__(self,user_num,bs_antenna_num,power_constr,noise_variance,layer_num,mode,batch_size):
        super(unfold_net, self).__init__()
        self.user_num=user_num
        self.bs_antenna_num=bs_antenna_num
        self.layer_num=layer_num
        self.power_constr=power_constr
        self.noise_variance=noise_variance
        self.proportion=[0.1,0.9]
        self.batch_size=batch_size
        self.norm=normalized_P_1RS(user_num,bs_antenna_num,power_constr)
        self.l1=unfold_layer(self.bs_antenna_num,self.user_num,self.power_constr,self.noise_variance,self.batch_size)
        self.l2=unfold_layer(self.bs_antenna_num,self.user_num,self.power_constr,self.noise_variance,self.batch_size)
        self.l3=unfold_layer(self.bs_antenna_num,self.user_num,self.power_constr,self.noise_variance,self.batch_size)
        self.l4=unfold_layer(self.bs_antenna_num,self.user_num,self.power_constr,self.noise_variance,self.batch_size)
        self.l5=unfold_layer(self.bs_antenna_num,self.user_num,self.power_constr,self.noise_variance,self.batch_size)

    def count_SR(self,H,P):
        batch_size=H.size(dim=0)
        H=torch.permute(H,(1,2,0))
        P=torch.permute(P,(1,2,0))
        predict_sumrate = torch.zeros((1, batch_size))
        P_c,P_p=P[:,-1,:][:,np.newaxis],P[:,0:user_num,:]
        for sample_index in range(batch_size):
            com_rate=torch.zeros(self.user_num)
            for user_index in range(self.user_num):
                sig = torch.square(torch.norm(P_p[:, user_index, sample_index].matmul(
                    torch.conj(H[:,user_index,sample_index]))))
                interf = - sig + torch.square(torch.norm(
                    torch.conj(H[:, user_index, sample_index]).matmul(P_p[:, :, sample_index])))+self.noise_variance
                com_interf=sig+interf
                com_sig=torch.square(torch.norm(P_c[:, 0, sample_index].matmul(
                    torch.conj(H[:,user_index,sample_index]))))
                com_rate[user_index]=torch.log2(1+com_sig/com_interf)
                predict_sumrate[0, sample_index] = predict_sumrate[0, sample_index] + torch.log2(1 + sig / interf)
            predict_sumrate[0, sample_index]+=torch.min(com_rate)
        return -torch.sum(predict_sumrate)/batch_size
    def init_P(self, H):
        Pt = self.power_constr
        N_user = self.user_num
        P_p, P_c = self.proportion[0] * Pt / N_user, self.proportion[1] * Pt
        P_batch = torch.zeros(H.shape[0], self.bs_antenna_num, self.user_num + 1, dtype=torch.cfloat,device=device)
        for iter in range(H.shape[0]):
            P = []
            for i in range(N_user):
                P.append(H[iter, :, i][:, np.newaxis] / torch.norm(H[iter, :, i]) * (P_p**0.5))
            P = torch.cat(P, dim=1)
            a, _, _ = torch.linalg.svd(H[iter])
            p_c =(P_c**0.5) * a[:, 0]
            P_batch[iter, :, 0:self.user_num] = P
            P_batch[iter, :, -1] = p_c
        return P_batch

    def found_order(self, H):
        H_strength = []
        diction = dict()
        for i in range(self.user_num):
            h = torch.norm(H[:, i])
            H_strength.append(h)
            diction[h] = i
        H_strength2 = copy.deepcopy(H_strength)
        sum_1 = sum(H_strength)
        H_strength3 = []
        for h_strength in H_strength:
            H_strength3.append(h_strength / sum_1)
        H_strength.sort()
        H_strength.reverse()
        H_strength3.sort()
        herms = []
        for h in H_strength:
            herms.append(diction[h])
        for j in range(len(herms)):
            H_strength2[herms[j]] = H_strength3[j]
        return H_strength2

    def init_lambda(self, H):
        norm_H=torch.norm(H,dim=1)
        _, order = torch.sort(norm_H, descending=True,dim=1)
        lambda_i=(norm_H/torch.sum(norm_H,dim=1).unsqueeze(1))
        lambda_i = lambda_i ** 2 / torch.sum(lambda_i ** 2,dim=1).unsqueeze(1)
        mu_batch=(torch.ones((H.shape[0],1),device=device)*(10/self.power_constr)).float()
        torch.autograd.set_detect_anomaly(True)
        return torch.cat((lambda_i,mu_batch),dim=1).float()

    def forward(self,H):
        precoders_init=self.init_P(H)
        parameters_init=self.init_lambda(H)
        precoders1,parameters1=self.l1(precoders_init,H,parameters_init)
        precoders1=self.norm.normalize_P(precoders1,self.batch_size)
        precoders2,parameters2=self.l2(precoders1,H,parameters1)
        precoders2=self.norm.normalize_P(precoders2,self.batch_size)
        precoders3,parameters3=self.l3(precoders2,H,parameters2)
        precoders3=self.norm.normalize_P(precoders3,self.batch_size)
        precoders4,parameters4=self.l4(precoders3,H,parameters3)
        precoders4=self.norm.normalize_P(precoders4,self.batch_size)
        precoders5,parameters5=self.l5(precoders4,H,parameters4)
        precoders5=self.norm.normalize_P(precoders5,self.batch_size)
        return precoders5,torch.stack([parameters1,parameters5],2)

class base_net(nn.Module):
    def __init__(self,user_num):
        super(base_net, self).__init__()
        self.fc1 = nn.Linear((user_num+(user_num*2+1)*bs_antenna_num*2+1), 512)
        self.fc2=nn.Linear(512,(user_num+1))
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x=nn.functional.relu(x)
        x = self.fc2(x)
        x=self.sigmoid(x)
        return x

class Net(nn.Module):
    def __init__(self,user_num,bs_antenna_num,power_constr,mode):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(8)
        if mode=='DBL_SR_SDMA':
            self.fc1 = nn.Linear(8*user_num*2*bs_antenna_num, user_num*bs_antenna_num*2)
            self.normalize = normalized_P_SDMA(bs_antenna_num, user_num,power_constr)
        elif mode=='BNN_SR_SDMA' or mode=='BNN_SR_SDMA_TEST':
            self.fc1 = nn.Linear(8 * user_num * 2 * bs_antenna_num, user_num * 2)
        elif mode=='DBL_SR_1RS':
            self.fc1 = nn.Linear(8*user_num*2*bs_antenna_num, (user_num+1)*bs_antenna_num*2)
            self.normalize = normalized_P_1RS(bs_antenna_num, user_num,power_constr)
        elif mode=='BNN_SR_1RS':
            self.fc1 = nn.Linear(8*user_num*2*bs_antenna_num, user_num * 4)
        self.sigmoid = nn.Sigmoid()
        self.mode=mode

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = nn.functional.dropout(x, p=0.3)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = nn.functional.dropout(x, p=0.3)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        if self.mode.find('DBL')!=-1:
            pass
            x=self.normalize(x)
        else:
            x = self.sigmoid(x)
        return x

class EarlyStopping:
    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model,now):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model,now)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model,now)
            self.counter = 0

    def save_checkpoint(self, val_loss, model,now):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss

def analyse_data(filename,mode):
    data = sio.loadmat(filename)
    sample_num = int(data['sample_num'])
    noise_variance = data['noise_variance'][0][0]
    bs_antenna_num = int(data['bs_antenna_num'])
    sum_rate=data['sumrate']
    user_num = int(data['user_num'])

    power_constr = data['power_constr'][0][0]
    sample_channel_complex = data['channel_bs_user_complex']
    beam = data['beam']
    config=dict()
    config['power_constr']=power_constr
    config['noise_variance']=noise_variance
    config['user_num']=user_num
    config['bs_antenna_num']=bs_antenna_num
    config['sum_rate']=sum_rate
    config['sample_num']=sample_num
    config['times']=data["times"]
    sample_channel = np.zeros((sample_num, 1, user_num * bs_antenna_num, 2))
    for sample_index in range(sample_num):
        sample_channel[sample_index, 0, :, 0] = sample_channel_complex[:, :, sample_index].reshape(-1,user_num * bs_antenna_num).real.astype('float32')
        sample_channel[sample_index, 0, :, 1] = sample_channel_complex[:, :, sample_index].reshape(-1,user_num * bs_antenna_num).imag.astype('float32')
    if mode=='unfold_SR_1RS':
        x = torch.tensor(np.transpose(sample_channel_complex,(2,0,1))).type(torch.complex64)
        y=torch.tensor(np.transpose(beam,(2,0,1)))
    else:
        x =torch.tensor(sample_channel).float()
        if mode=='DBL_SR_SDMA':
            sample_beam = np.zeros((sample_num, user_num * bs_antenna_num * 2))
            for sample_index in range(sample_num):
                temp=beam[:,: , sample_index]
                temp2 = power_constr / np.trace(temp @ np.conj(temp.T))
                beam[:, :, sample_index]=np.sqrt(temp2)*temp
                sample_beam[sample_index, :user_num * bs_antenna_num] = beam[:, :, sample_index].reshape(-1,user_num * bs_antenna_num).real.astype('float32')
                sample_beam[sample_index, user_num * bs_antenna_num:] = beam[:, :, sample_index].reshape(-1,user_num * bs_antenna_num).imag.astype('float32')
            y=torch.tensor(sample_beam).float()
        elif mode=='DBL_SR_1RS':
            sample_beam = np.zeros((sample_num, (user_num+1) * bs_antenna_num * 2))
            for sample_index in range(sample_num):
                temp=beam[:,: , sample_index]
                temp2 = power_constr / np.trace(temp @ np.conj(temp.T))
                beam[:, :, sample_index]=np.sqrt(temp2)*temp
                sample_beam[sample_index, :(user_num+1)  * bs_antenna_num] = beam[:, :, sample_index].reshape(-1,(user_num+1)  * bs_antenna_num).real.astype('float32')
                sample_beam[sample_index, (user_num+1)  * bs_antenna_num:] = beam[:, :, sample_index].reshape(-1,(user_num+1)  * bs_antenna_num).imag.astype('float32')
            y=torch.tensor(sample_beam).float()
        elif mode=='unfold_SR_1RS_supervised':
            x = torch.tensor(np.transpose(sample_channel_complex, (2, 0, 1))).type(torch.complex64)
            y=torch.tensor(data['langrage'],dtype=torch.float32, device=device)
        else:
            print("wrong mode")
            exit(0)
    return x,y,config

def split_train_val(x,y,batch_size,shuffle=True):
    dataset = TensorDataset(x, y)
    val_percent = 0.2
    val_size = int(val_percent * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=shuffle)
    return train_dataloader,val_dataloader

def check_dataset(train_config,test_config):
    if train_config['user_num']!=test_config['user_num']:
        exit(0)
    if train_config['bs_antenna_num']!=test_config['bs_antenna_num']:
        exit(0)
    user_num=train_config['user_num']
    bs_antenna_num=train_config['bs_antenna_num']
    return user_num,bs_antenna_num

def batch_test(test_x, test_y,batch_size,shuffle,model,criterion_SR):
    dataset = TensorDataset(test_x, test_y)
    test_loader=DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    test_loss=0
    test_SR=0
    total=0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            [outputs,L1,L2] = model(batch_x)
            batch_SR = -criterion_SR(batch_x, outputs)
            batch_loss=criterion_MSE(L1,L2,batch_y)
            test_loss += batch_loss.item() * batch_x.size(0)
            test_SR+=batch_SR.item() * batch_x.size(0)
            total += batch_y.size(0)
    avg_test_loss = test_loss / total
    print("Test Loss:"+str(avg_test_loss))
    print("Test sum rate:"+str(test_SR))

def save_checkpoint(model,optimizer,epoch,now,file_name):
    checkpoint = {
        "net": model.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": epoch,
        "now":now
    }
    try:
        torch.save(checkpoint, 'checkpoints/' + file_name + '/model_' + now + '.pth')
    except:
        os.makedirs('checkpoints/' + file_name)
        torch.save(checkpoint, 'checkpoints/' + file_name + '/model_' + now + '.pth')

def save_log(now,config,renew):
    try:
       f=open("./log/"+file_name+"/"+now+"log.txt","a")
       if renew==True:
           f.write(json.dumps(config))
    except:
        os.makedirs('log/' + file_name)
        f = open("./log/" + file_name + "/" + now + "log.txt", "a")
        if renew == True:
            f.write(json.dumps(config))
    return f


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    config=dict()

    config['test_file_name']='data/RSMA22_test.mat'
    config['train_file_name']='data/RSMA22_train.mat'
    config['seed']=20
    setup_seed(config['seed'])

    file_name, extension = os.path.splitext(os.path.basename(__file__))
    now = datetime.datetime.now()
    now = now.strftime('%Y%m%d%H%M%S')
    config['mode']='unfold_SR_1RS_supervised'
    config['batch_size']=1000
    x,y,train_config=analyse_data(config['train_file_name'],config['mode'])
    test_x,test_y,test_config=analyse_data(config['test_file_name'],config['mode'])
    user_num,bs_antenna_num=check_dataset(train_config, test_config)
    config['noise_variance'],config['user_num'],config['bs_antenna_num'],config['power_constr'],config['sample_num_train'],config['sample_num_test']=train_config['noise_variance'],train_config['user_num'],train_config['bs_antenna_num'],train_config['power_constr'],train_config['sample_num'],test_config['sample_num']
    train_dataloader, val_dataloader=split_train_val(x,y,config['batch_size'],True)
    ave_real_sumrate=np.sum(test_config['sum_rate'])/test_config['sample_num']
    if config['mode']=='unfold_SR_1RS' or config['mode']=='unfold_SR_1RS_supervised':
        model=unfold_net(user_num,bs_antenna_num,train_config['power_constr'],train_config['noise_variance'],5,config['mode'],config['batch_size'])
    if config['mode']=='unfold_SR_1RS_supervised':
        criterion_MSE=loss_unfold_SR_1RS_supervised()
        criterion_SR=loss_unfold_SR_1RS(bs_antenna_num,user_num,train_config['power_constr'],train_config['noise_variance'])
    config['lr']=0.0001
    config['annotation']='demo_for_Nt2K2_RSBNN'
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0.)
    # ---------------- Start Training ------------------
    save_path='checkpoints\\'+file_name
    early_stopping = EarlyStopping(save_path)
    renew = True
    if renew == True:
        start_epoch = 0
        print("renew start")
        log_file = save_log(now, config,renew)
    else:
        lists = os.listdir(save_path)
        lists.sort(key=lambda x: os.path.getmtime((save_path + "\\" + x)))
        file_new = os.path.join(save_path, lists[-1])
        checkpoint = torch.load(file_new)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        now=checkpoint['now']
        log_file = save_log(now, config,renew)
    print('Model Summary', model)
    print('Model Summary', model,file=log_file)
    config['num_epochs'] = 200
    config['cut']=50
    train_SR=0
    train_size=0
    train_losses=[]
    val_losses=[]
    for epoch in range(start_epoch,config['num_epochs']):
        save_checkpoint(model, optimizer, epoch, now,file_name)
        model.train()
        train_loss = 0.0
        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            [outputs, L] = model(inputs)
            train_size+=1
            if epoch>=config['cut']:
                loss=criterion_SR(inputs,outputs)
            else:
                loss = criterion_MSE(L,labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_dataloader.dataset)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(val_dataloader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                [outputs, L]  = model(inputs)
                if epoch>=config['cut']:
                    loss = criterion_SR(inputs, outputs)
                else:
                    loss = criterion_MSE(L, labels)
                val_loss += loss.item() * inputs.size(0)
            val_loss /= len(val_dataloader.dataset)
        print('Epoch ',epoch,'Train Loss:',train_loss,', Val Loss:',val_loss)
        print('Epoch ',epoch,'Train Loss:',train_loss,', Val Loss:',val_loss,file=log_file)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        early_stopping(val_loss,model,now)
        if early_stopping.early_stop and epoch>=config['cut']:
            print("Early stopping")
            break
    [outputs, L] = model(test_x.to(device))
    ave_real_sumrate=np.sum(test_config['sum_rate'])/test_config['sample_num']
    dataset = TensorDataset(test_x, test_y)
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    model_cpu=model.to("cpu")
    t_total=0
    t_count=0
    SR_total=0
    device = "cpu"
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            inputs, labels = data
            inputs = inputs.to('cpu')
            labels = labels.to('cpu')
            T1=time.time()
            [outputs, L] = model_cpu(inputs)
            test_loss = criterion_SR(inputs.to(device), outputs.to(device))
            T2=time.time()
            t_total+=(T2-T1)
            t_count+=1
            SR_total+=test_loss.item()
    print("test average time",t_total/t_count,"FP average time",np.sum(test_config['times'])/test_config['sample_num'])
    print("test average SR",SR_total/t_count,'FP average SR:', ave_real_sumrate)
    print("test average time", t_total / t_count,"FP average time",np.sum(test_config['times'])/test_config['sample_num'],file=log_file)
    print("test average SR", -SR_total / t_count,'FP average SR:', ave_real_sumrate,file=log_file)
