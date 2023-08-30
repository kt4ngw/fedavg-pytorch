import argparse


def input_options():
    parser = argparse.ArgumentParser()
    # iid
    parser.add_argument('-is_iid', type=bool, default=True, help='data distribution is iid.')
    parser.add_argument('--dataset_name', type=str, default='mnist_dir_0.1', help='name of dataset.')
    parser.add_argument('--model_name', type=str, default='mnist_cnn', help='the model to train')
    parser.add_argument('--gpu', type=bool, default=True, help='gpu id to use(e.g. 0,1,2,3)')
    parser.add_argument('--round_num', type=int, default=301, help='number of round in comm')
    parser.add_argument('--num_of_clients', type=int, default=100, help='numer of the clients')
    parser.add_argument('--c_fraction', type=float, default=0.1,
                        help='C fraction, 0 means 1 client, 1 means total clients')
    parser.add_argument('--local_epoch', type=int, default=5, help='local train epoch')
    parser.add_argument('--batch_size', type=int, default=32, help='local train batch size')
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate, \
                        use value from origin paper as default")
    parser.add_argument('--gn0', type=int, default=1, help='gno')
    parser.add_argument('--seed', help='seed for randomness;', type=int, default=3001)
    parser.add_argument('--weight_decay', help='weight_decay;', type=int, default=1)
    args = parser.parse_args()
    options = args.__dict__

    return options
