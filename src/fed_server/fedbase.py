import numpy as np
import torch
import time
from src.fed_client.client import BaseClient
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import copy
from src.utils.metrics import Metrics
import torch.nn.functional as F
criterion = F.cross_entropy


class BaseFederated(object):
    def __init__(self, options, dataset, clients_label, model=None, optimizer=None, name=''):
        if model is not None and optimizer is not None:
            self.model = model
            self.optimizer = optimizer
        self.options = options
        self.dataset = dataset
        self.clients_label = clients_label
        self.gpu = options['gpu']
        self.batch_size = options['batch_size']
        self.num_round = options['round_num']
        self.per_round_c_fraction = options['c_fraction']
        self.clients = self.setup_clients(self.dataset, self.clients_label)
        self.clients_num = len(self.clients)
        self.name = '_'.join([name, f'wn{int(self.per_round_c_fraction * self.clients_num)}',
                              f'tn{len(self.clients)}'])
        self.metrics = Metrics(options, self.clients, self.name)
        self.latest_global_model = self.get_model_parameters()


    @staticmethod
    def move_model_to_gpu(model, options):
        if options['gpu'] is True:
            device = 0
            torch.cuda.set_device(device)
            # torch.backends.cudnn.enabled = True
            model.cuda()
            print('>>> Use gpu on device {}'.format(device))
        else:
            print('>>> Don not use gpu')

    def get_model_parameters(self):
        state_dict = self.model.state_dict()
        return state_dict

    def set_model_parameters(self, model_parameters_dict):
        state_dict = self.model.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = model_parameters_dict[key]
        self.model.load_state_dict(state_dict)

    def train(self):
        """The whole training procedure

        No returns. All results all be saved.
        """
        raise NotImplementedError

    def setup_clients(self, dataset, clients_label):
        train_data = dataset.train_data
        train_label = dataset.train_label
        all_client = []
        for i in range(len(clients_label)):
            local_client = BaseClient(self.options, i, TensorDataset(torch.tensor(train_data[self.clients_label[i]]),
                                                torch.tensor(train_label[self.clients_label[i]])), self.model, self.optimizer)
            all_client.append(local_client)

        return all_client

    def local_train(self, round_i, select_clients, ):

        local_model_paras_set = []
        stats = []
        for i, client in enumerate(select_clients, start=1):
            client.set_model_parameters(self.latest_global_model)
            local_model_paras, stat = client.local_train()
            local_model_paras_set.append(local_model_paras)
            stats.append(stat)
            if True:
                print("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
                      "Loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s ".format(
                       round_i, client.id, i, int(self.per_round_c_fraction * self.clients_num),
                       stat['loss'], stat['acc'] * 100, stat['time'], ))
        return local_model_paras_set, stats



    def aggregate_parameters(self, local_model_paras_set):

        averaged_paras = copy.deepcopy(self.model.state_dict())
        train_data_num = 0
        for var in averaged_paras:
            averaged_paras[var] = 0
        for num_sample, local_model_paras in local_model_paras_set:
            for var in averaged_paras:
                averaged_paras[var] += num_sample * local_model_paras[var]
            train_data_num += num_sample
        for var in averaged_paras:
            averaged_paras[var] /= train_data_num
        return averaged_paras



    def test_latest_model_on_testdata(self, round_i):
        # Collect stats from total test data
        begin_time = time.time()
        stats_from_test_data = self.global_test(use_test_data=True)
        end_time = time.time()

        if True:
            print('= Test = round: {} / acc: {:.3%} / '
                  'loss: {:.4f} / Time: {:.2f}s'.format(
                   round_i, stats_from_test_data['acc'],
                   stats_from_test_data['loss'], end_time-begin_time))
            print('=' * 102 + "\n")

        self.metrics.update_test_stats(round_i, stats_from_test_data)

    def global_test(self, use_test_data=True):
        assert self.latest_global_model is not None
        self.set_model_parameters(self.latest_global_model)
        test_data = self.dataset.test_data
        test_label = self.dataset.test_label
        print("testLabel", test_label)
        testDataLoader = DataLoader(TensorDataset(torch.tensor(test_data), torch.tensor(test_label)), batch_size=10, shuffle=False)
        test_loss = test_acc = test_total = 0.
        with torch.no_grad():
            for X, y in testDataLoader:
                if self.gpu:
                    X, y = X.cuda(), y.cuda()
                    pred = self.model(X)
                    loss = criterion(pred, y)
                    _, predicted = torch.max(pred, 1)

                    correct = predicted.eq(y).sum()
                    test_acc += correct.item()
                    test_loss += loss.item() * y.size(0)
                    test_total += y.size(0)

        stats = {'acc': test_acc / test_total,
                 'loss': test_loss / test_total,
                 'num_samples': test_total,}
        return stats





