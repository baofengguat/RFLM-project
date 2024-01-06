from Model.Check_model import *
from Code.DataLoading import ServerLoading
import copy

def FedInit(args):
    train_batches, test_batches,A = ServerLoading(args)
    server_model = []
    w_server = []
    w_local = [[]] * len(train_batches)

    for server in train_batches:
        Number = len(server.dataset.files)
        if Number < 50:
            model = Check_model('mobilenet')
            server_model.append(model)
            nParams = sum([p.nelement() for p in model.parameters()])
            print('Number of mobilenet parameters is:%d'% nParams)
            w_server.append(model.state_dict())

        else:
            model = Check_model('ResNet18')
            server_model.append(model)
            nParams = sum([p.nelement() for p in model.parameters()])
            print('Number of ResNet18 parameters is:%d' % nParams)
            w_server.append(model.state_dict())
    global_model = copy.deepcopy(w_server)
    personalized_model = copy.deepcopy(w_server)


    return server_model, train_batches, test_batches,A,\
           w_server,w_local,global_model,personalized_model


