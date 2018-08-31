import os
import argparse

class opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
        self.parser.add_argument('-num_epoch', type=int, default=300, help='training epochs')
        self.parser.add_argument('-batch_size', type=int, default=100, help='training batch size')
        self.parser.add_argument('-nRow', type=int, default=30, help='how many rows of images in the output')
        self.parser.add_argument('-nCol', type=int, default=30, help='how many columns of images in the output')
        self.parser.add_argument('-img_size', type=int, default=64, help='output image size')

        #test setting
        self.parser.add_argument('-test_size', type=int, default=1, help='How many images to generate during testing')
        self.parser.add_argument('-test', action = 'store_true', help='add `-test` for testing')
        self.parser.add_argument('-score', action = 'store_true', help='add `-score` for reporting scores')

        self.parser.add_argument('-z_size', type=int, default=100, help='dimension of latent variable sample from latent space')
        self.parser.add_argument('-category', default='alp', help='training category')
        self.parser.add_argument('-data_path', default='./data/scene', help='path to data')
        self.parser.add_argument('-output_dir', default='./result_images', help='directory to save output synthesized images')
        self.parser.add_argument('-log_dir', default='./checkpoint', help='directory to save logs')
        self.parser.add_argument('-ckpt_dir', default='./checkpoint', help='directory to save checkpoints')
        self.parser.add_argument('-log_epoch', type=int, default=50, help='save checkpoint each `log_epoch` epochs')

        self.parser.add_argument('-set', default='scene', help='which dataset, scene/cifar')
        self.parser.add_argument('-with_noise', type=bool, default=False, help='add noise during the langevin or not')
        self.parser.add_argument('-incep_interval', type=int, default=0, help='intervals to compute inception score. 0 for not computing')

        #Generator Parameters
        self.parser.add_argument('-ckpt_des', default=None, help='load checkpoint for descriptor')
        self.parser.add_argument('-sigma_gen', type=float, default=0.3,help='sigma of reference distribution')
        self.parser.add_argument('-langevin_step_num_gen', type=int, default=0, help='langevin step number for generator')
        self.parser.add_argument('-langevin_step_size_gen', type=float, default=0.1, help='langevin step size for generator')
        self.parser.add_argument('-lr_gen', type=float, default=0.0001,help='learning rate of generator')
        self.parser.add_argument('-beta1_gen', type=float, default=0.5,help='beta of Adam for generator')

        #Descriptor Parameters
        self.parser.add_argument('-ckpt_gen', default=None, help='load checkpoint for generator')
        self.parser.add_argument('-sigma_des', type=float, default=0.016, help='sigma of reference distribution')
        self.parser.add_argument('-langevin_step_num_des', type=int, default=10, help='langevin step number for descriptor')
        self.parser.add_argument('-langevin_step_size_des', type=float, default=0.001, help='langevin step size for descriptor')
        self.parser.add_argument('-lr_des', type=float, default=0.01,help='learning rate of descriptor')
        self.parser.add_argument('-beta1_des', type=float, default=0.5,help='beta of Adam for descriptor')


    def parse(self):
        self.init()
        self.opt = self.parser.parse_args()

        args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
                    if not name.startswith('_'))
        if not os.path.exists(self.opt.ckpt_dir):
            os.makedirs(self.opt.ckpt_dir)
        file_name = os.path.join(self.opt.ckpt_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('==> Args:\n')
            for k, v in sorted(args.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))
        return self.opt