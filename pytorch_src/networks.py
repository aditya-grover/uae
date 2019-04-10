import torch
import torch.nn as nn
import torch.nn.functional as F



class EncoderDecoder(nn.Module):
    def __init__(self, 
                nin, 
                args):
        super(EncoderDecoder, self).__init__()
        self.loss_type = args.loss_type
        self.linear = args.linear
        self.zdim = args.zdim
        self.args = args

        enc_nout = self.zdim
        if self.loss_type == 'vae' or self.loss_type == 'bvae':
            enc_nout = 2*self.zdim 

        if self.linear:
            enc_net = [nn.Linear(nin, enc_nout)]
        else:
            enc_net = [nn.Linear(nin, 500),
                        nn.ReLU(),
                        nn.Linear(500, enc_nout)]

        self.enc_net = nn.Sequential(*enc_net)

        self.dec_net = nn.Sequential(nn.Linear(self.zdim, 500),
                                    nn.ReLU(),
                                    nn.Linear(500, 500),
                                    nn.ReLU(),
                                    nn.Linear(500, nin),
                                    nn.Sigmoid())

    def encode(self, x):

        x = x.view(x.size(0), -1) #flatten tensor
        z_param = self.enc_net(x)

        return z_param

    def decode(self, z):

        x = self.dec_net(z)

        return x

    def forward(self, x):

        loss_type = self.loss_type

        if loss_type == 'dae':
            x = x + torch.randn_like(x) * self.args.sigma

        z_param = self.encode(x)

        mu = None
        logvar = None
        if loss_type == 'uae':
            mu = z_param
            std = self.args.sigma
            eps = torch.randn_like(mu)
            z = eps.mul(std).add_(mu)
        elif loss_type == 'ae' or loss_type == 'dae':
            z = z_param
        elif loss_type == 'vae' or loss_type == 'bvae':
            mu = z_param[:, :self.zdim]
            logvar = z_param[:, self.zdim:]
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            z =  eps.mul(std).add_(mu)

        xhat = self.decode(z)

        return xhat, mu, logvar

        













