from src.options import input_options
from src.utils.tools import get_each_client_data_index
from getdata import GetDataSet
from src.fed_server.fedavg import FedAvgTrainer
def main():
    options = input_options()
    dataset = GetDataSet("MNIST")
    each_client_label_index = get_each_client_data_index(dataset.train_label, options["num_of_clients"])
    FedAvg = FedAvgTrainer(options, dataset, each_client_label_index)
    FedAvg.train()

if __name__ == '__main__':
    main()




