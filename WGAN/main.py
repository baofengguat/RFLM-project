from utils.config import parse_args
from utils.data_loader import get_data_loader

from models.wgan_gradient_penalty import WGAN_GP

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main(args):
    model = WGAN_GP  #None
    if args.model == 'WGAN-GP':
        model = WGAN_GP(args)
    else:
        print("Model type non-existing. Try again.")
        exit(-1)

    # Load datasets to train and test loaders
    train_loader, test_loader = get_data_loader(args)

    # Start model training
    if args.is_train == 'True':
        model.train(train_loader)
    else:
        model.evaluate(test_loader, args.load_D, args.load_G,args.save_path,args.number)


if __name__ == '__main__':
    args = parse_args()
    print(args.channels)
    print(args.cuda)
    main(args)
