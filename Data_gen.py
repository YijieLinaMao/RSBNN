import random
import scipy.io as sio
import time
import cvxpy as cvx
import numpy as np
import math
import copy
import torch
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
np.random.seed(0)
random.seed(0)

class algorithm_fp_torch(torch.nn.Module):
    def __init__(self):
        super(algorithm_fp_torch, self).__init__()
        self.rho = 0.1
        self.tolerance = 1e-4
        self.toleranceinner = 1e-5
        self.maxcount = 1000
        self.maxcountinner = 1000
        self.proportion = [0.1, 0.9]

    def found_order(self,H,N_user):
        H_strength = []
        diction = dict()
        for i in range(N_user):
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
        # H_strength3.reverse()
        herms = []
        for h in H_strength:
            herms.append(diction[h])
        for j in herms:
            H_strength2[herms[j]] = H_strength3[j]
        return H_strength2
    def init_lambda(self, H, N_user):
        lambda_i = self.found_order(H, N_user)
        lambda_i = [abs(l_i) ** 2 for l_i in lambda_i]
        sum_lambda = sum(lambda_i)
        lambda_i = [l_i / sum_lambda for l_i in lambda_i]
        lambda_i = torch.Tensor(lambda_i).view(N_user, 1)
        return lambda_i

    def init_P(self, Pt, H, N_user):
        P_p, P_c = self.proportion[0] * Pt / N_user, self.proportion[1] * Pt
        P = []
        for i in range(N_user):
            P.append(H[:, i].unsqueeze(1) / torch.norm(H[:, i]) *(P_p**0.5))
        P = torch.cat(P, dim=1)
        U, S, V = torch.svd(H)
        p_c = (P_c**0.5) * U[:, 0].unsqueeze(1)
        return p_c, P

    def count_SR(self, H, Nt, N_user, Pt):
        flag = 1
        obj_past = 0
        count = 0
        noise = torch.ones((N_user, 1))
        p_c, P = self.init_P(Pt, H, N_user)
        lambda_i = self.init_lambda(H, N_user)
        mu = 10 / Pt
        iter_least = 5
        iter_start = 1
        Langrage_list = []
        while (flag == 1):
            T_k = self.count_T_k(H, P, noise)
            T_ck = T_k + torch.abs(torch.mm(torch.conj(H.T), p_c)) ** 2
            alpha_ck, alpha_k = self.count_alpha_k(H, P, p_c, N_user, T_k)
            beta_k = torch.sqrt(1 + alpha_k) * torch.reshape(torch.diag(np.conj(H.T) @ P), (N_user, 1)) / T_k
            beta_ck = torch.sqrt(1 + alpha_ck) * torch.mm(torch.conj(H.T), p_c) / T_ck
            res_k = torch.log2(1 + alpha_k) - alpha_k
            res_ck = torch.log2(1 + alpha_ck) - alpha_ck
            count_inner = 0
            while (count_inner <= self.maxcountinner):
                H_ck, H_k = self.update_H(beta_k, beta_ck, lambda_i, H)
                p_c, P = self.update_P(H_ck, H_k, mu, Nt, lambda_i, alpha_ck, alpha_k, beta_ck, beta_k, H, N_user)
                T_kcvx, T_ckcvx = self.count_T(H, P, p_c, noise, N_user)
                E_ck = self.count_E_ck(beta_ck, T_ckcvx, alpha_ck, H, p_c, res_ck)
                mu = torch.real(torch.trace(torch.mm(P, torch.conj(P.T)) + torch.mm(p_c, torch.conj(p_c.T))) / Pt) * mu
                rate_c = E_ck - torch.min(E_ck) + self.rho + torch.abs(torch.min(E_ck))
                y = torch.min(rate_c)
                index = torch.argmin(rate_c)
                lambda_i = (y / rate_c) * lambda_i
                lambda_out = 1 - torch.sum(lambda_i)
                lambda_i[index] = lambda_out + lambda_i[index]
                lambda_i = torch.abs(lambda_i)
                e = torch.abs(torch.trace(torch.mm(P, torch.conj(P.T)) + torch.mm(p_c, torch.conj(p_c.T))) / Pt - 1) * mu + torch.abs(lambda_out)
                if torch.norm(e) < self.toleranceinner:
                    break
                count_inner = count_inner + 1
            count += 1
            iter_start += 1
            H_ck, H_k = self.update_H(beta_k, beta_ck, lambda_i, H)
            p_c, P = self.update_P(H_ck, H_k, mu, Nt, lambda_i, alpha_ck, alpha_k, beta_ck, beta_k, H, N_user)
            T_kcvx, T_ckcvx = self.count_T(H, P, p_c, noise, N_user)
            E_kcvx = self.count_E_k(beta_k, T_kcvx, alpha_k, H, P, res_k, N_user)
            E_ckcvx = self.count_E_ck(beta_ck, T_ckcvx, alpha_ck, H, p_c, res_ck)
            obj = torch.sum(E_kcvx) + torch.min(E_ckcvx)
            if torch.abs(obj - obj_past) <= self.tolerance and count != 1:
                flag = 0
            else:
                obj_past = obj
            if iter_start == 2 or flag == 0 or count >= self.maxcount:
                lamb_list = list(lambda_i.view(-1))
                lamb_list.append(mu)
                Langrage = torch.Tensor(lamb_list).view(-1, 1)
                Langrage_list.append(Langrage)
            if count >= self.maxcount:
                break
        Y = self.count_sum_rate(H, P, p_c, noise, N_user)
        Langrage_list = torch.cat(Langrage_list, dim=1)
        return Y.item(), torch.cat([P, p_c], dim=1), Langrage_list

    def count_T_k(self, H, P, noise):
        HP = torch.square(torch.abs(torch.mm(torch.conj(H.T), P)))
        T_k = torch.sum(HP, dim=1, keepdim=True)
        return T_k + noise

    def count_sum_rate(self, H, P, p_c, noise, N_user):
        T_ktest, T_cktest = self.count_T(H, P, p_c, noise, N_user)
        rate_c, rate_p = self.count_rate(T_cktest, T_ktest, H, P, N_user)
        Y = torch.sum(rate_p) + torch.min(rate_c)
        return Y

    def count_T(self, H, P, p_c, noise, K):
        T_kcvx = self.count_T_k(H, P, noise)
        square_abs_Hp_c = torch.square(torch.abs(torch.mm(torch.conj(H.T), p_c)))
        square_abs_Hp_c = square_abs_Hp_c.view(K, 1)
        T_ckcvx = T_kcvx + square_abs_Hp_c
        return T_kcvx, T_ckcvx

    def count_alpha_k(self, H, P, p_c, K, T_k):
        abs_square = torch.square(torch.abs(torch.diag(torch.conj(H.T) @ P)))
        alpha_k = T_k / (T_k - abs_square.view(K, 1)) - 1
        alpha_ck = torch.square(torch.abs(torch.mm(torch.conj(H.T), p_c))) / T_k
        return alpha_ck, alpha_k

    def update_H(self, beta_k, beta_ck, lambda_i, H):
        beta_k_square_abs = torch.square(torch.abs(torch.conj(beta_k).T))
        beta_ck_square_abs = torch.square(torch.abs(torch.conj(beta_ck).T))
        lambda_i_H = torch.conj(lambda_i).T
        H_k = torch.sqrt(beta_k_square_abs + lambda_i_H * beta_ck_square_abs) * H
        H_ck = torch.sqrt(lambda_i_H * beta_ck_square_abs) * H
        return H_ck, H_k

    def update_P(self, H_ck, H_k, mu, Nt, lambda_i, alpha_ck, alpha_k, beta_ck, beta_k, H, N_user):
        p_c_one = torch.mm(H_ck, torch.conj(H_ck.T)) + mu * torch.eye(Nt)
        p_c_two = torch.conj(lambda_i.T) * torch.sqrt(1 + torch.conj(alpha_ck.T)) *beta_ck.T* H
        p_c_two = torch.sum(p_c_two, dim=1)
        P_one = torch.mm(H_k, torch.conj(H_k.T)) + mu * torch.eye(Nt)
        P_two = (torch.sqrt(1 + alpha_k) * beta_k).T * H
        return torch.mm(torch.inverse(p_c_one), p_c_two.view(Nt, 1)), torch.mm(torch.inverse(P_one), P_two)

    def count_E_ck(self, beta_ck, T_ckcvx, alpha_ck, H, p_c, res_ck):
        square_abs_beta_ck = torch.square(torch.abs(beta_ck))
        real_part = torch.real(torch.conj(beta_ck) * (torch.conj(H.T) @ p_c))
        E_ck = -square_abs_beta_ck * T_ckcvx + 2 * torch.sqrt(1 + alpha_ck) * real_part + res_ck
        return E_ck

    def count_E_k(self, beta_k, T_kcvx, alpha_k, H, P, res_k, N_user):
        square_abs_beta_k = torch.square(torch.abs(beta_k))
        diag = torch.diag(torch.conj(H.T) @ P)
        diag = diag.view(N_user, 1)
        real_part = torch.real(torch.conj(beta_k) * diag)
        E_ck = -square_abs_beta_k * T_kcvx + 2 * torch.sqrt(1 + alpha_k) * real_part + res_k
        return E_ck

    def count_abcd(self, alpha_k, beta_k, beta_ck, alpha_ck, lam, mu):
        a = torch.square(torch.abs(beta_k)) / mu
        b = lam * torch.square(torch.abs(beta_k)) / mu
        c = torch.abs(mu * torch.sqrt(1 + alpha_k) * beta_k) ** 2
        d = torch.abs(mu * torch.sqrt(1 + alpha_ck) * beta_ck * lam) ** 2
        return a, b, c, d

    def count_rate(self, T_cktest, T_ktest, H, P, K):
        rate_c = torch.log2(T_cktest / T_ktest)
        HP_diag = torch.diag(torch.conj(H.T) @ P)
        HP_diag = HP_diag.view(K, 1)
        HP_square_abs = torch.square(torch.abs(HP_diag))
        rate_p = torch.log2(T_ktest / (T_ktest - HP_square_abs))
        return rate_c, rate_p
class algorithm_WMMSE():
    def __init__(self):
        self.tolerance=1e-4
        self.proportion=[0.1,0.9]
        self.rth=0

    def rec(self,m, n):
        return math.factorial(n) // (math.factorial(m) * math.factorial(n - m))

    def count_set_number(self,n):
        sum = 0
        for i in range(n):
            sum += self.rec(i + 1, n)
        return sum

    def abs_square_of_Hk_mul_Pi(self,H, P, k, i):
        try:
            HP = abs(np.matmul(H[:, k].conjugate().T[np.newaxis, :], P[:, i][:, np.newaxis])[0, 0]) ** 2
        except:
            HP = self.abs_square_of_Hk_mul_Pi_cvx(H, P, k, i)
        return HP

    def Pi_mul_Hk(self,H, P, k, i):
        return np.matmul(P[:, i].conjugate().T[np.newaxis, :], H[:, k][:, np.newaxis])[0, 0]

    def count_g_i(self,abs_square_H_P, t):
        return abs_square_H_P * (t ** (-1))

    def count_epsilon(self,t, i):
        return (t ** (-1)) * i

    def count_u(self,e):
        return e ** (-1)

    def count_ksai_i_cvx(self,e, ui):
        return ui * e - np.log2(ui)

    def count_ksai_cvx(self,epsilon, u,Nr):
        ksai = [[], []]
        for k in range(Nr):
            ksai_12 = self.count_ksai_i_cvx(epsilon[k][0], u[k][0])
            ksai_1 = self.count_ksai_i_cvx(epsilon[k][1], u[k][1])
            ksai[k].append(ksai_12)
            ksai[k].append(ksai_1)
        return ksai

    def abs_square_of_Hk_mul_Pi_cvx(self,H, P, k, i):
        Hk_H = H[:, k].conjugate().T[np.newaxis, :]
        Pi = P[:, i][:, np.newaxis]
        return cvx.square(cvx.abs(cvx.matmul(Hk_H, Pi)))

    def count_e_cvx(self,g, H, P, k, i, t):
        square_1 = (abs(g)) ** 2
        real_1 = cvx.real(g * (cvx.matmul(H[:, k].conjugate().T[np.newaxis, :], P[:, i][:, np.newaxis])))
        e = square_1 * t - 2 * real_1 + 1
        return e

    def Init_weight_k(self,number):
        weight = []
        for i in range(number):
            weight.append(1)
        return weight

    def Init_H_g(self,gamma, theta, Nr, Nt):
        h1 = np.ones((Nt, 1)) - 1j * np.zeros((Nt, 1))
        for j in range(Nr - 1):
            h = np.ones((Nt, 1)) - 1j * np.zeros((Nt, 1))
            for i in range(Nt):
                if i == 0:
                    h[i, 0] = gamma[j] * h[i, 0]
                    continue
                h[i, 0] = gamma[j] * np.cos(theta * (j + 1) * i) - 1j * gamma[j] * np.sin(theta * (j + 1) * i)
            h1 = np.concatenate((h1, h), axis=1)
        return h1

    def Init_P_g(self,H, Pt, Nr):
        P = []
        power = Pt * self.proportion[-1]
        a, b, c = np.linalg.svd(H)
        P.append(np.sqrt(power) * a[:, 0][:, np.newaxis])
        power_pri = Pt * self.proportion[0] / Nr
        for i in range(Nr):
            P.append(H[:, i][:, np.newaxis] / np.linalg.norm(H[:, i]) * np.sqrt(power_pri))
        return np.concatenate(P, axis=1)

    def count_T_MMSE_g(self,H, P, Nr):
        T = []
        for k in range(Nr):
            T_k = []
            T_private_k = 1
            for i in range(Nr):
                T_private_k += self.abs_square_of_Hk_mul_Pi(H, P, k, 1 + i)
            T_comm_k = T_private_k + self.abs_square_of_Hk_mul_Pi(H, P, k, 0)
            T_k.append(T_comm_k)
            T_k.append(T_private_k)
            T.append(T_k)
        return T

    def count_I_MMSE_g(self,T, H, P, Nr):
        I = []
        for k in range(Nr):
            I_k = T[k][1:len(T[k]) + 1]
            I_private_k = 1
            for i in range(Nr):
                if i != k:
                    I_private_k += self.abs_square_of_Hk_mul_Pi(H, P, k, 1 + i)
            I_k.append(I_private_k)
            I.append(I_k)
        return I

    def count_g_MMSE_g(self,H, P, T, Nr):
        g = []
        for k in range(Nr):
            g_k = []
            g_k.append(self.Pi_mul_Hk(H, P, k, 0) * (T[k][0] ** (-1)))
            g_k.append(self.count_g_i(self.Pi_mul_Hk(H, P, k, 1 + k), T[k][1]))
            g.append(g_k)
        return g

    def count_eplison_MMSE_g(self,T, I, Nr):
        eplison = []
        for k in range(Nr):
            eplison_k = []
            for i in range(len(T[k])):
                eplison_k.append(self.count_epsilon(T[k][i], I[k][i]))
            eplison.append(eplison_k)
        return eplison

    def count_u_MMSE_g(self,eplison, Nr):
        u = []
        for k in range(Nr):
            u_k = []
            for i in range(len(eplison[k])):
                u_k.append(self.count_u(eplison[k][i]))
            u.append(u_k)
        return u

    def count_ksai_g(self,u, eplison, Nr):
        ksai = []
        for k in range(Nr):
            ksai_k = []
            for i in range(len(eplison[k])):
                ksai_k.append(self.count_ksai_i_cvx(eplison[k][i], u[k][i]))
            ksai.append(ksai_k)
        return ksai

    def count_ksai_tot_g(self,ksai, x, Nr):
        ksai_tot = []
        for k in range(Nr):
            ksai_tot_k = ksai[k][-1]
            ksai_tot_k += x[k][0]
            ksai_tot.append(ksai_tot_k)
        return ksai_tot

    def count_eplison_cvx_g(self,T, g, H, P, Nr):
        eplison = []
        for k in range(Nr):
            eplison_k = []
            eplison_k.append(self.count_e_cvx(g[k][0], H, P, k, 0, T[k][0]))
            eplison_k.append(self.count_e_cvx(g[k][-1], H, P, k, 1 + k, T[k][-1]))
            eplison.append(eplison_k)
        return eplison

    def ksai_comm_constraint(self,x, ksai, Nr):
        constraints = []
        x_comm_sum = 1
        for k in range(Nr):
            x_comm_sum += x[k][0]
        for i in range(Nr):
            constraints.append(ksai[i][0] <= x_comm_sum)
        return constraints

    def optimize_P_x_g(self,u, g, H, Pt, miu_weight_WSR, Nr, Nt):
        P_opt = cvx.Variable((Nt, Nr + 1), complex=True)
        x = cvx.Variable((Nr, 1))
        T = self.count_T_MMSE_g(H, P_opt, Nr)
        epsilon = self.count_eplison_cvx_g(T, g, H, P_opt, Nr)
        ksai = self.count_ksai_g(u, epsilon, Nr)
        ksai_tot = self.count_ksai_tot_g(ksai, x, Nr)
        sum_P = 0 + 0j
        for i in range(Nr + 1):
            sum_P += cvx.quad_form(P_opt[:, i][:, np.newaxis], np.eye(Nt))
        constraint1 = self.ksai_comm_constraint(x, ksai, Nr)
        constraint2 = cvx.real(sum_P) <= Pt
        constraint3 = []
        for k in range(Nr):
            constraint3.append(ksai_tot[k] <= 1 - self.rth)
        constraint4 = x <= np.zeros((Nr, 1))
        constraints = []
        constraints.append(constraint4)
        constraints = constraints + constraint3
        constraints.append(constraint2)
        constraints = constraints + constraint1
        equation = 0
        for k in range(Nr):
            equation += miu_weight_WSR[k] * ksai_tot[k]
        obj = cvx.Minimize(equation)
        prob = cvx.Problem(obj, constraints)
        prob.solve(solver=cvx.MOSEK)
        return P_opt.value, x.value, obj.value

    def count_rate(self,T, I, Nr):
        rate = []
        for k in range(Nr):
            rate.append(np.log2(T[k][-1] / I[k][-1]))
        return rate

    def count_rate_tot(self,rate, x, Nr):
        rate_tot = []
        for k in range(Nr):
            rate_tot_k = rate[k]
            rate_tot_k -= x[k][0]
            rate_tot.append(rate_tot_k)
        return rate_tot

    def count_sum_rate(self,miu_weight_WSR, rate_tot, Nr):
        sum_rate = 0
        for i in range(Nr):
            sum_rate += miu_weight_WSR[i] * rate_tot[i]
        return sum_rate

    def P_init(self,P_p, H, K):
        P = []
        for i in range(K):
            P.append(H[:, i][:, np.newaxis] / np.linalg.norm(H[:, i]) * np.sqrt(P_p))
        return P

    def P_total_init(self,Pt, H, K):
        P_c = Pt * 0.9
        P_p = (Pt - P_c) / K
        a, b, c = np.linalg.svd(H)
        p_c = np.sqrt(P_c) * a[:, 0][:, np.newaxis]
        P = self.P_init(P_p, H, K)
        P.append(p_c)
        return np.concatenate(P, axis=1)

    def count_SR(self,miu_weight_WSR, H, Pt, Nr, Nt):
        sum = [0]
        L = []
        P = self.Init_P_g(H, Pt, Nr)
        n = 0
        while True:
            n += 1
            T = self.count_T_MMSE_g(H, P, Nr)
            I = self.count_I_MMSE_g(T, H, P, Nr)
            e = self.count_eplison_MMSE_g(T, I, Nr)
            g = self.count_g_MMSE_g(H, P, T, Nr)
            u = self.count_u_MMSE_g(e, Nr)

            P, x, obj = self.optimize_P_x_g(u, g, H, Pt, miu_weight_WSR, Nr, Nt)
            sum.append(obj)
            if abs(sum[n] - sum[n - 1]) < self.tolerance or n >= 2000:
                T = self.count_T_MMSE_g(H, P, Nr)
                I = self.count_I_MMSE_g(T, H, P, Nr)
                rates = self.count_rate(T, I, Nr)
                rate_tot = self.count_rate_tot(rates, x, Nr)
                sum_rate = self.count_sum_rate(miu_weight_WSR, rate_tot, Nr)
                L.append(sum_rate)
                break
        return sum_rate, P

class DataSave_Base():
    def __init__(self,filename,train_size,test_size,Nt,K,SNR):
        self.partition=0.25
        self.train_size=int(train_size)
        self.test_size=int(test_size)
        self.Nt=Nt
        self.K=K
        self.SNR=SNR
        self.Pt=10**(self.SNR/10)
        self.Pt_list=[]
        self.H_list=[]
        self.SR_list = []
        self.P_list = []
        self.filename=filename
        self.times=[]

    def data_gen(self,H_org):
        self.count(H_org)
        train_dict=self.make_dict("train")
        test_dict=self.make_dict("test")
        sio.savemat(".\data\\"+self.filename+"_train.mat", train_dict)
        sio.savemat(".\data\\"+self.filename+"_test.mat", test_dict)
    def make_dict(self,mode):
        if mode=="train":
            sample_num=self.train_size
            index=[i for i in range(self.test_size,self.test_size+self.train_size)]
        elif mode=="test":
            sample_num=self.test_size
            index=[i for i in range(0,self.test_size)]
        else:
            print("wrong mode: the mode must be train or test")
            exit(0)
        one = np.ones((1, 1))
        data = dict()
        data=self.key_var(data,index)
        data['beam'] = self.P_list[:, :, index]
        data['sumrate'] = self.SR_list[:, index]
        data['power_constr'] = one * self.Pt
        data['noise_variance'] = one
        data['sample_num'] = sample_num
        data['channel_bs_user_complex'] = self.H_list[:, :, index]
        data['user_num'] = self.K
        data['bs_antenna_num'] = self.Nt
        data['times']=self.times[:,index]
        if mode=="test":
            print("time:",np.average(data['times']))
            print("SR:",np.average(data['sumrate']))
        return data


class generate_H():
    def __init__(self,d0,path_loss_exponent,dk_up=None,dk_down=0):
        self.d0=d0
        self.dk_up=dk_up
        self.dk_down=dk_down
        self.path_loss_exponent=path_loss_exponent
    def H_gen(self,train_size,test_size,Nt,K):
        H_org=[]
        for i in range(train_size + test_size):
            np.random.seed(i)
            H=self.gen_one_H(i,Nt,K)
            H_org.append(H)
        return H_org
    def gen_one_H(self,seed,Nt,K):
        np.random.seed(seed)
        H = 1 / np.sqrt(2) * (np.random.randn(Nt, K) + 1j * np.random.randn(Nt, K))
        dk=np.random.randint(self.dk_down, high=self.dk_up, size=K, dtype='l')
        rho=1/(1+(dk/self.d0)**self.path_loss_exponent)
        H=rho*H
        return H


class DataSave_RSMA_torch(DataSave_Base):
    def __init__(self,filename,train_size,test_size,Nt,K,SNR):
        super().__init__(filename,train_size,test_size,Nt,K,SNR)
        self.alg=algorithm_fp_torch()
        self.Langrages_list=[]
    def count(self,H_org):
        t_total=0
        self.times=[]
        for i in range(len(H_org)):
            #print("iter",i)
            H=H_org[i]
            HH=torch.tensor(H)
            T1=time.time()
            SR, precoders ,Langrages= self.alg.count_SR(HH, self.Nt, self.K, self.Pt)
            T2=time.time()
            t_total+=(T2-T1)
            self.times.append(T2-T1)
            precoders=np.array(precoders)
            Langrages=np.array(Langrages[ np.newaxis,:,:])
            self.SR_list.append(SR)
            self.P_list.append(precoders[:, :, np.newaxis])
            self.H_list.append(H[:,:,np.newaxis])
            self.Langrages_list.append(Langrages)
        self.SR_list=np.array(self.SR_list)[np.newaxis]
        self.times=np.array(self.times)[np.newaxis]
        self.P_list=np.concatenate(self.P_list,axis=2)
        self.H_list=np.concatenate(self.H_list,axis=2)
        self.Langrages_list=np.concatenate(self.Langrages_list,axis=0)
        self.times=np.concatenate(self.times)[np.newaxis]
    def key_var(self,data,index):
        data['langrage']=self.Langrages_list[index,:,:]

        return data
class DataSave_RSMA_WMMSE(DataSave_Base):
    def __init__(self,filename,train_size,test_size,Nt,K,SNR):
        super().__init__(filename,train_size,test_size,Nt,K,SNR)
        self.alg=algorithm_WMMSE()
    def count(self,H_org):
        t_total=0
        SR_total=0
        iter=0
        for i in range(len(H_org)):
            H=H_org[i]
            T1=time.time()
            if i<self.test_size:
                iter+=1
                SR,precoders= self.alg.count_SR([1 for i in range(self.K)], H, self.Pt, self.K, self.Nt)
            else:
                SR=0
                precoders=np.zeros((self.Nt,self.K+1))
            SR_total+=SR
            T2=time.time()
            t_total+=(T2-T1)
            self.SR_list.append(SR)
            self.P_list.append(precoders[:, :, np.newaxis])
            self.H_list.append(H[:,:,np.newaxis])
            self.times.append(T2-T1)
        self.SR_list=np.array(self.SR_list)[np.newaxis]
        self.P_list=np.concatenate(self.P_list,axis=2)
        self.H_list=np.concatenate(self.H_list,axis=2)
        self.times=np.array(self.times)[np.newaxis]
        self.times=np.concatenate(self.times)[np.newaxis]

    def key_var(self,data,index):
        return data


if __name__ =="__main__":
    Nt=2
    K=2
    SNR=20
    train_size=20000
    test_size=100

    H_generator= generate_H(30, 3, 100)
    H=H_generator.H_gen(train_size,test_size,Nt,K)
    print("FP-HFPI:")
    datasave=DataSave_RSMA_torch("demo_RSMA22",train_size,test_size,Nt,K,SNR)
    datasave.data_gen(H)
    print("WMMSE:")
    datasave_WMMSE=DataSave_RSMA_WMMSE("demo_RSMA_WMMSE_22",train_size,test_size,Nt,K,SNR)
    datasave_WMMSE.data_gen(H)