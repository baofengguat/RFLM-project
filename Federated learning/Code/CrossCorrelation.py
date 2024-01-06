from Code.DataLoading import *
from config import load_config
import torch.nn.functional as F

class CommonInformation:
    def __init__(self, args, dataloader, model,global_param):
        self.args = args
        self.dataloader = dataloader #list
        self.global_param = global_param #list
        self.model = model

    def run(self):
        sum_value = []
        for id in range(len(self.global_param)):
            model = self.model[id]
            model.to(self.args.device)
            model.load_state_dict(self.global_param[id])
            sum_value.append(self.client_run(model,self.dataloader))

        value_metrix = self.information_merge(sum_value)
        print('------------------Common—vadation-completed------------------')
        return value_metrix

    def client_run(self,model,dataloader):
        value_id = []
        for batch_idx, (image, label) in enumerate(dataloader):
            if self.args.CUDA == True:
                image = image.cuda()
                label = label.cuda()
            output,_ = model(image)
            preds, pred = output.data.cpu().topk(1, dim=1)
            value_id.append(preds.cpu().squeeze().detach().numpy())
        return value_id

    def information_merge(self,sum_value):
        s = []
        for list in sum_value:
            k = [j for i in list for j in i]
            s.append(torch.tensor(k))
        value_metrix = torch.stack(s,0)
        return value_metrix
















args = load_config()
common_data = Common_Loading(args)
