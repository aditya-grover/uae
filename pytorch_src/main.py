from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image
import os, sys
from networks import *
from dataset import *

def reconstruction_loss(recon_x, x, **kwargs):

    MSE = F.mse_loss(recon_x, x.view(x.shape[0], -1), reduction='sum')

    return MSE

def vae_loss(recon_x, x, **kwargs):

    kwargs = kwargs['kwargs']
    mu = kwargs['mu']
    logvar = kwargs['logvar']

    MSE = reconstruction_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD

def beta_vae_loss(recon_x, x, **kwargs):

    kwargs = kwargs['kwargs']
    mu = kwargs['mu']
    logvar = kwargs['logvar']
    beta = kwargs['beta']

    MSE = reconstruction_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + beta * KLD

def get_loss_function(loss_type):

    if loss_type == 'vae':
        return vae_loss
    elif loss_type == 'bvae':
        return beta_vae_loss
    else:
        return reconstruction_loss


def train(args, model, device, train_loader, optimizer, epoch, loss_function):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        kwargs = {'mu': mu, 'logvar': logvar, 'beta': args.beta}
        loss = loss_function(recon_batch, data, kwargs=kwargs)
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

    
def eval(args, model, device, heldout_loader, loss_function, validation=True):
    model.eval()
    heldout_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(heldout_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            kwargs = {'mu': mu, 'logvar': logvar, 'beta': args.beta}
            heldout_loss += loss_function(recon_batch, data, kwargs=kwargs).item()

    heldout_loss /= len(heldout_loader.dataset)
    if validation:
        print('====> Heldout set loss: {:.4f}'.format(heldout_loss))
    n = min(data.size(0), 8)
    comparison = torch.cat([data[:n], recon_batch.view(args.test_batch_size, 1, 28, 28)[:n]])

    return heldout_loss, comparison


def main():

    parser = argparse.ArgumentParser(description='PyTorch implementation of UAE')
    
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
    parser.add_argument('--loss-type', type=str, default='uae',
                        help='Options: ae/uae/vae/dae/wae/bvae')
    parser.add_argument('--zdim', type=int, default=50,
                        help='number of latent dimensions')
    parser.add_argument('--sigma', type=float, default=0.01, 
                        help='Noise/bandwidth hyperparameter for uae/wae/dae')
    parser.add_argument('--beta', type=float, default=1.5,
                        help='beta hyperparameter for bvae')
    parser.add_argument('--linear', dest='linear', action='store_true', default=False,
                        help='uses linear encoder if true')
    parser.add_argument('--dump', dest='dump', action='store_true', default=False,
                        help='dump')

    args = parser.parse_args()
    args.datadir = os.path.join(args.datadir, args.dataset)
    is_linear = 'linear' if args.linear else 'nonlinear'
    exp_logdir = os.path.join(args.logdir, args.dataset, is_linear, str(args.zdim), args.loss_type, args.exp_id)
    resdir = os.path.join(exp_logdir, 'results')
    ckptdir = os.path.join(exp_logdir, 'ckpts')

    if not os.path.exists(ckptdir):
        os.makedirs(ckptdir)
    if not os.path.exists(resdir):
        os.makedirs(resdir)

    import json
    with open(os.path.join(resdir, 'config.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4, separators=(',', ': '))

    if args.dump:
        dump_file = os.path.join(resdir, 'log.txt')
        if not args.train: # append to existing log file
            sys.stdout = open(dump_file, 'a')
        else: # write a new log file
            sys.stdout = open(dump_file, 'w')

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

    if args.loss_type == 'bvae':
        loss_function = beta_vae_loss
    elif args.loss_type == 'vae':
        loss_function = vae_loss
    else:
        loss_function = reconstruction_loss

    model = EncoderDecoder(nin=nin, args=args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    savefile = os.path.join(ckptdir, 'best.pt')

    if args.train:
        best_validation_loss = np.inf
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch, loss_function)
            validation_loss, comparison = eval(args, model, device, valid_loader, loss_function, validation=True)
            
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save(model.state_dict(), savefile)
                save_image(comparison.cpu(), os.path.join(resdir, 'reconstruction_best_valid.png'), nrow=comparison.shape[0])

    state_dict = torch.load(savefile)
    model.load_state_dict(state_dict)
    test_loss, comparison = eval(args, model, device, test_loader, loss_function, validation=False)
    print('Test set loss: {:.4f}'.format(test_loss))
    save_image(comparison.cpu(), os.path.join(resdir, 'reconstruction_best_test.png'), nrow=comparison.shape[0])
    


if __name__ == '__main__':
    main()


