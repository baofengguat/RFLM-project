from Code.BuildServer import FedInit
from config import load_config
import copy
from Code.Federated import *
import numpy as np
from Code.util import *
from Code.aggregator import parameter_aggregate,acc_aggregator,read_out
from Code.DataLoading import ServerLoading,Common_Loading
from Code.CrossCorrelation import CommonInformation
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def ModelSave(args,NameList,personalized_model,global_model):
    os.makedirs(args.Model_save,exist_ok=True)
    torch.save(global_model, os.path.join(args.Model_save, 'Global.pth'))
    for Index,Name in enumerate(NameList):
        torch.save(personalized_model[Index], os.path.join(args.Model_save, '%s.pth'%Name))

def main(args):

    server_model, train_batches, test_batches, A,\
    w_server,w_local,global_model,personalized_model= FedInit(args)
    # Representative data loading
    common_data = Common_Loading(args)

    best_auc = 0.

    print("Start Training...")
    for com in range(1, args.com_round + 1):
        selected_user = np.random.choice(range(len(server_model)), len(server_model), replace=False)
        train_loss = []
        train_acc = {}
        for id in selected_user:
            engine = FedEngine(args,copy.deepcopy(train_batches[id]),server_model[id],global_model[id],personalized_model[id],w_local[id],{},id,"Train")
            outputs = engine.run()

            w_server[id] = copy.deepcopy(outputs['params'])
            w_local[id] = copy.deepcopy(outputs['params'])
            train_loss.append(outputs["loss"])
            train_acc[id]=(outputs["acc"])

        mtrain_loss = np.mean(train_loss)
        mtrain_acc = np.mean(list(train_acc.values()))

        log = 'Communication Round: {:03d}, Train Loss: {:.4f}, Train Accuracy: {:.4f}/com_round'
        print2file(log.format(com, mtrain_loss, mtrain_acc),args.logDir, True)

        # Server aggregation
        str1 = CommonInformation(args, common_data, server_model,global_model)
        value_metrix = str1.run()

        t1 = time.time()
        personalized_model = parameter_aggregate(args, A, w_server, global_model,value_metrix)
        #personalized_model = parameter_aggregate(args, A, w_server, global_model, None)
        t2 = time.time()
        log = 'Communication Round: {:03d}, Aggregation Time: {:.4f} secs'
        print2file(log.format(com, (t2 - t1)), args.logDir, True)

        global_model = read_out(personalized_model,args.device,dp=args.dp)
        #global_model = acc_aggregator(personalized_model,train_acc,dp=args.dp)

        # Validation
        if com % args.valid_freq == 0:
            batch_acc_train = {}
            batch_acc_test = {}

            batch_auc_train = {}
            batch_auc_test = {}

            train_batchesn, test_batchesn, _ = ServerLoading(args,False)

            for id in selected_user:
                # Robust model validation train
                tengine = FedEngine(args,copy.deepcopy(train_batchesn[id]),
                                server_model[id],personalized_model[id],personalized_model[id],w_local[id],{},id,"Test")
                outputs_train = tengine.Validation('train')
                # Robust model validation test
                tengine = FedEngine(args, copy.deepcopy(test_batchesn[id]),
                                    server_model[id],  personalized_model[id], personalized_model[id],w_local[id], {}, id,
                                    "Test")
                outputs_test = tengine.Validation('Test')

                # batch_loss.append(outputs["loss"])
                # batch_acc_train[outputs_train["c_state"]]= outputs_train["auc"]
                # batch_acc_test[outputs_test["c_state"]] = outputs_test["auc"]
                #
                # batch_auc_train[outputs_train["c_state"]] =  outputs_train["AucList"]
                # batch_auc_test[outputs_test["c_state"]] = outputs_test["AucList"]

                batch_acc_train[outputs_train["c_state"]]= outputs_train["acc"]
                batch_acc_test[outputs_test["c_state"]] = outputs_test["acc"]

            if np.array(list(batch_acc_test.values())).mean() > best_auc:
                best_auc = np.array(list(batch_acc_test.values())).mean()
                ModelSave(args,batch_acc_train.keys(),personalized_model,global_model)
                AucResuluts_logs(args, True, batch_auc_train.values(), batch_auc_train.keys())
                AucResuluts_logs(args, False, batch_auc_test.values(), batch_auc_test.keys())
                print("Model saved---Average ACC:%.4f" % best_auc)

            print("The ACC of each center train was:",batch_acc_train)
            print("The ACC of each center test was:",batch_acc_test)

if __name__ == "__main__":

    args = load_config()
    main(args)
