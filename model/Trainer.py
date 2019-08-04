from util.ops_normal import *
from torch import optim
import torch
from tensorboardX import SummaryWriter
import shutil
import torchvision as tv
import torch.nn
import torch.nn.functional as F
import time
from ops.img_ops import *
from skimage.color import rgb2lab, lab2rgb

class Trainer(object):
    def __init__(self, model, name="optimizer"):
        self.config = model.config
        self.model = model
        self.name = name
        self.optimizer = None
        self.summary_writer = None # will add class for summary writer soon

        if self.config.OPTIMIZER_METHOD == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE, betas=(0.9, 0.999))

    def print_model_param(self):
        for param_tensor in self.optimizer.state_dict():
            print(param_tensor, self.optimizer.state_dict()[param_tensor])

    def epoch_init(self):
        # deterministic initialization, and SUMMARY rewrite
        seed_everything(self.config.SEED + self.config.EPOCH)
        self.config.EPOCH += 1

        if self.config.SUMMARY:
            model_name = get_original_model_name(self.config.MODEL_NAME)
            summary_model_dir = self.config.SUMMARY_ROOT + "/" + model_name
            summary_dir = summary_model_dir + "/" + "EPOCH_{:04}".format(self.config.EPOCH)
            if os.path.exists(summary_dir):
                shutil.rmtree(summary_dir)
            self.summary_writer = SummaryWriter(summary_dir)


class ColorizationTrainer(Trainer):
    def __init__(self, model, data_pool, name="Colorization_Optimizer"):
        super(ColorizationTrainer, self).__init__(model, name)
        self.data_pool = data_pool

    def add_summary(self, step, loss, lab_norm, output):
        self.summary_writer.add_scalars('loss', {'L1_loss': loss}, step)
        if step % self.config.SUMMARY_IMG_ITER is 0:
            lab_norm[:,1:,:,:] = output
            lab = self.model.lab_unnormal(lab_norm)
            lab = img_tensor_to_numpy(lab)

            rgb = [lab2rgb(lab[i]) for i in range(self.config.BATCH_SIZE)]
            rgb = np.asarray(rgb)
            rgb = img_numpy_to_tensor(rgb)

            grid_output = tv.utils.make_grid(rgb, normalize=True, scale_each=True)
            self.summary_writer.add_image('output', grid_output, step)

        if step % self.config.PRINT_ITER is 0:
            print("[TRAINING] Epoch[{}]({}/{}): Loss: {:.4f} Time: {:.2f}".format(self.config.EPOCH, step + 1, len(self.data_pool),
                                                                     loss, time.time()-self.stime))

    def training(self):
        model = self.model.cuda()
        if self.config.CUDA:
            model = torch.nn.DataParallel(self.model).cuda()

        self.model.load()
        self.stime = time.time()
        while self.config.OBJ_EPOCH > self.config.EPOCH:
            model.train()
            self.epoch_init()
            print('[TRAINING] Epoch[{}] Start ---'.format(self.config.EPOCH))

            for step, batch in enumerate(self.data_pool):
                self.optimizer.zero_grad()

                lab = batch['lab'].cuda()
                hint = batch['hint'].cuda()
                lab_norm = self.model.lab_normal(batch['lab']).cuda()
                ab_norm = lab_norm[:,1:,:,:]

                l = lab[:,:1,:,:]
                ab = lab[:,1:,:,:]

                out = model(l,ab,hint)
                loss = F.smooth_l1_loss(out, ab_norm)

                loss.backward()
                self.optimizer.step()
                self.config.STEP += 1
                self.add_summary(step+1, loss.item(), lab_norm.detach().cpu(), out.detach().cpu())

            self.model.save()
            self.summary_writer.close()

