from model.model import CoopNets
from opts import opts

def main():
    opt=opts().parse()
    model=CoopNets(opt)
    if opt.test or opt.test_inception:
        model.test()
    else:
        model.train()

if __name__=='__main__':
    main()
