import torch
import datetime

def print2file(buf, out_file, p=False):
    if p:
        print(buf)
    outfd = open(out_file, 'a+')
    outfd.write(str(datetime.datetime.now()) + '\t' + buf + '\n')
    outfd.close()

def sd_matrixing(state_dic):
    """
    Turn state dic into a vector
    :param state_dic:
    :return:
    """
    keys = []
    param_vector = None
    for key, param in state_dic.items():
        keys.append(key)
        if param_vector is None:
            param_vector = param.clone().detach().flatten().cpu()
        else:
            if len(list(param.size())) == 0:
                param_vector = torch.cat((param_vector, param.clone().detach().view(1).cpu().type(torch.float32)), 0)
            else:
                param_vector = torch.cat((param_vector, param.clone().detach().flatten().cpu()), 0)
    return param_vector