# RFLM Code Usage Instructions

## Federated learning
The main part of RFLM algorithm, including federated learning framework, robust learning of GCN and WGAN.
- Analysis of robustness : Common and adaptive feature calculation section
- Code : Body code section
- Model : Model base file

Method of Use :
    Go to the Config.py file
    
    - -- model：Select model 
    - -- original-dir: The path where the data is entered
    - -- common-data: Represents the path of the data
    - -- category: Two types of labels
    - -- Model-save: Model storage path
    - -- txt-dir: The txt text path of the patient order was recorded
    - -- clients : Number of clients
    - -- com_round: Number of federated cycles
    - -- client_epochs: Number of local client runs
    
```shell
   python /Code/data_deal_with.py
   python /Code/main.py
```   
    
## WGAN 
WGAN algorithm part, used to generate fake images of the corresponding dataset and used as representative dataset.
- models : The main part of the WGAN algorithm
- utils : Data loading and other supplementary files
- main.py : Main function

Method of Use :
   ```shell
   python main.py --model WGAN-GP --is_train True --download False --dataroot datasets/CT-data 
   --dataset CTDATA --generator_iters 40000 --cuda True --batch_size 16
   ``` 

   Description of parameters:
   
   - --model：WGAN-GP
   - --is_train：Whether it is a training phase
   - --download：Whether the dataset needs to be downloaded
   - --dataroot：The directory where the training data is located. 
                 You can set it to the root directory where your dataset is located
   - --dataset：Which data set to use. You can set it to your own dataset
   - --generator_iters：Number of training rounds
   - --cuda：Whether to use GPU
   
Notes:
    Only one type of sample can be generated at a time, if you need to generate samples 
      of more than one type, you need to train several times to get multiple generators.
 

## Contact information

The software enviroment for the study is complex and the related version information can be found in the files for the manuscript. 
If any problem, you are welcome to contact me with [here](jmlws2@163.com). Especially, I also will be happy to discuss the related 
research for this study or any other topic regarding the cancer for human beings. Looking forward to hearing from you.

Sincerely

Wansheng Long    
