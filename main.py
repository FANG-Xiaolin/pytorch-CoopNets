import torch
from model.model import CoopNets
from opts import opts

def main():
    opt=opts().parse()
    model=CoopNets(opt)
    model.train()
    if opt.set=='cifar':
        opt.img_size=32

if __name__=='__main__':
    main()
