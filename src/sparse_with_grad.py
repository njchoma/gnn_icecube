import torch

class SPMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, x):
        ctx.save_for_backward(A, x)
        return torch.mm(A, x)

    @staticmethod
    def backward(ctx, grad_input):
        A, x = ctx.saved_tensors
        if A.is_sparse:
            i = A._indices()
            v = (torch.mul(grad_input[i[0]], x[i[1]])).sum(dim=1)
            t_type = torch.cuda if A.is_cuda else torch
            grad_0 = t_type.sparse.FloatTensor(i,v,A.size())
        else:
            grad_0 = torch.mm(grad_input, x.t())
        return grad_0, torch.mm(A.t(), grad_input)

class Construct_Adj(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i, v, size):
        ctx.nb_val = v.size(0)
        t_type = torch.cuda if v.is_cuda else torch
        sp_adj = t_type.sparse.FloatTensor(i, v, size)
        return sp_adj

    @staticmethod
    def backward(ctx, grad_input):
        if grad_input.is_cuda:
            # Account for lack of sum in cuda
            v = grad_input._values()
            nb_layers = int(v.size(0) / ctx.nb_val)
            v = v.reshape(nb_layers, -1).sum(dim=0)
            i = grad_input._indices()[0].reshape(nb_layers, -1)
        else:
            v = grad_input._values()
        return None, v, None
