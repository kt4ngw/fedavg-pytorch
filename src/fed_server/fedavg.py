from src.fed_server.fedbase import BaseFederated
from torch import optim
from src.optimizers.adam import MyAdam
import numpy as np
from src.models.models import choose_model

class FedAvgTrainer(BaseFederated):
    def __init__(self, options, dataset, clients_label):
        model = choose_model(options)
        self.move_model_to_gpu(model, options)
        self.optimizer = MyAdam(model.parameters(), lr=options['lr'])
        super(FedAvgTrainer, self).__init__(options, dataset, clients_label, model, self.optimizer,)

    def train(self):
        print('>>> Select {} clients per round \n'.format(int(self.per_round_c_fraction * self.clients_num)))

        # self.latest_global_model = self.get_model_parameters()
        for round_i in range(self.num_round):

            self.test_latest_model_on_testdata(round_i)

            selected_clients = self.select_clients()

            local_model_paras_set, stats = self.local_train(round_i, selected_clients)


            self.latest_global_model = self.aggregate_parameters(local_model_paras_set)

            self.optimizer.adjust_learning_rate(round_i)

        self.test_latest_model_on_testdata(self.num_round)
        self.metrics.write()

    def select_clients(self):
        num_clients = min(int(self.per_round_c_fraction * self.clients_num), self.clients_num)
        index = np.random.choice(len(self.clients), num_clients, replace=False,)
        select_clients = []
        for i in index:
            select_clients.append(self.clients[i])
        return select_clients