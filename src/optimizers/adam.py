from torch.optim import Adam

class MyAdam(Adam):
    def __init__(self, params, lr,):
        self.lr = lr
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))

        super(MyAdam, self).__init__(params, lr)
    def adjust_learning_rate(self, round_i):

         for group in self.param_groups:
             group['lr'] = self.lr / (round_i + 1)