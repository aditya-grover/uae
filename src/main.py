from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import os, sys
from networks import *
from dataset import *


def train(args, model, device, train_loader, optimizer, epoch, P):
    model.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

    
def test(args, model, device, test_loader, validation=True):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.test_batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         os.path.join(resdir, 'reconstruction_' + str(epoch) + '.png'), nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

    return test_loss


def main():

    parser = argparse.ArgumentParser(description='PyTorch implementation of parametric t-sne')
    
    # training hyperparmeters
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, 
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='learning rate (default: 0.001)')

    # system parameters
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--train', dest='train', action='store_true', default=False,
                        help='trains a model from scratch if true')

    # file handling parameters
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='name of dataset')
    parser.add_argument('--logdir', type=str, default='./logs',
                        help='log directory')
    parser.add_argument('--exp-id', type=str, default='1')
    parser.add_argument('--datadir', type=str, default='../datasets')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='how many batches to wait before logging training status')

    # ablation parameters
    parser.add_argument('--loss', type=str, default='uae',
                        help='Options: ae/uae/vae/dae/wae/bvae')
    parser.add_argument('--zdim', type=int, default=50,
                        help='number of latent dimensions')
    parser.add_argument('--sigma', type=float, default=0.01, 
                        help='Noise/bandwidth hyperparameter for uae/wvae/dae')
    parser.add_argument('--beta', type=float, default=1.5,
                        help='beta hyperparameter for bvae')
    parser.add_argument('--linear', dest='train', action='store_true', default=False,
                        help='uses linear encoder if true')

    args = parser.parse_args()
    args.datadir = os.path.join(args.datadir, args.dataset)
    exp_logdir = os.path.join(args.logdir, args.dataset, args.exp_id)
    resdir = os.path.join(exp_logdir, 'results')
    ckptdir = os.path.join(exp_logdir, 'ckpts')

    if not os.path.exists(exp_logdir):
        os.makedirs(exp_logdir)
    if not os.path.exists(ckptdir):
        os.makedirs(ckptdir)
    if not os.path.exists(resdir):
        os.makedirs(resdir)

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader, valid_loader, test_loader = get_loaders(dataset=args.dataset,
                                            datadir=args.datadir,
                                            batch_size=args.batch_size,
                                            test_batch_size=args.test_batch_size,
                                            kwargs=kwargs)

    if args.dataset == 'mnist':
        nin = 784
    else:
        raise NotImplementedError

    enc_nout = zdim
    if args.loss == 'vae' or args.loss == 'bvae':
        enc_nout = 2*zdim 

    enc_module = LinearEncoder if args.linear else NonLinearEncoder
    encode = enc_module(nin=nin, nout=enc_nout)
    decode = Decoder(nin=zdim, nout=nin)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    savefile = os.path.join(ckptdir, 'best.pt')

    if args.train:
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch, P)
            # test(args, model, device, test_loader)
            torch.save(model.state_dict(), savefile)
    state_dict = torch.load(savefile)
    model.load_state_dict(state_dict)

if __name__ == '__main__':
    main()


