from util.ops_normal import *


class Trainer(object):
    def __init__(self, model):
        self.config = model.config
        self.model = model

    def save(self):
        print('hello world')

    def load(self):
        print('hello world')


class ColorizationTrainer(Trainer):
    def __init__(self, model, data_pool):
        super(ColorizationTrainer, self).__init__(model)
        self.data_pool = data_pool

    def training(self):
        while self.config.OBJ_EPOCH > self.config.EPOCH:
            self.model.train()
            seed_everything(self.config.SEED + self.config.EPOCH)

            for iter, batch in enumerate(self.data_pool):
                rgb = batch['rgb']
                # lab = batch['lab']
                # hint = batch['hint']
                # #
                # plt.imshow(np.concatenate((hint,hint,hint),axis=2))
                # plt.show()
