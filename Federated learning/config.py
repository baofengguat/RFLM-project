import argparse

def load_config():
    parser = argparse.ArgumentParser()

    #Model Config
    parser.add_argument('--model', type=str,choices=['Botnet18','ResNet18',
                                 'VGG16','densenet121','DBB_ResNet18','Swimtransfromer'],default='ResNet18',help='Select model')
    parser.add_argument('--blocktype', metavar='BLK', default='DBB', choices=['DBB', 'ACB', 'base'],help='Select DBB structure, not required for other models')

    # Config of FilePath
    parser.add_argument('--original-dir', type=str, default=r'Mutil_data', help='Raw data of an unpartitioned data set')
    parser.add_argument('--common-data', type=str, default=r'Representative_data', help='WG_DCGAN_CommonDATA')

    parser.add_argument('--logDir', type=str, default=r'..\logs', help='Root directory of the data')
    parser.add_argument('--category', type=list, default=['NR','R'], help='Two types of data labels')

    parser.add_argument('--train-save', type=str, default='..\\Auc_logs\\Validation B', help='Save the address during training')
    parser.add_argument('--Model-save', type=str, default='..\\Model_File\\Validation B', help='Save the address during training')
    parser.add_argument('--txt-dir', type=str, default='..\\txt\\Validation B', help='Save the address during training')
    parser.add_argument('--feature_save', type=str, default='..\\feature_map\\Validation B', help='Save the address during training')
    parser.add_argument('--clients', type=int, default=3, help="number of users: K")

    parser.add_argument('--agg', type=str, default='graph_common',
                        help='graph_v2，graph_v3，graph，scaffold，prox，scaf，scaffold，avg')
    parser.add_argument('--serverbeta', type=float, default=0.3, help='personalized agg rate alpha')

    parser.add_argument('--Node_relationship', type=str, default='wasserstein_distance',help="MI,cos,pairwise_distance,wasserstein_distance")

    # Graph Learning
    parser.add_argument('--subgraph_size', type=int, default=30, help='k')
    parser.add_argument('--adjalpha', type=float, default=3, help='adj alpha')
    parser.add_argument('--gc_epoch', type=int, default=5, help='')
    parser.add_argument('--adjbeta', type=float, default=0.05, help='update ratio')
    parser.add_argument('--edge_frac', type=float, default=1, help='the fraction of clients')
    parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')
    parser.add_argument('--serveralpha', type=float, default=1, help='server prop alpha')
    parser.add_argument('--layers', type=int, default=3, help='number of layers')

    # Config of model Training
    parser.add_argument('--numclass', type=int, default=2, help='How many categories to choose')
    parser.add_argument('--PreTrain', type=bool, default=True, help='Choose to use a natural image pretraining model')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
  
    parser.add_argument('--dp', type=float, default=0.)

    parser.add_argument('--seed', type=bool, default=False)
    parser.add_argument('--client_epochs', type=int, default=5)
    parser.add_argument('--CUDA', type=bool, default=True)
    parser.add_argument('--device', type=str, default='cuda', help='')
    parser.add_argument('--com_round', type=int, default=80, help='Number of communication round to train.')
    parser.add_argument('--valid_freq', type=int, default=1, help='validation at every n communication round')
    parser.add_argument('--feature_Number', type=int, default=4449, help='validation at every n communication round')

    return parser.parse_args()
