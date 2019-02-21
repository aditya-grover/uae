import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearEncoder(nn.Module):
	def __init__(self, nin, nout):
		super(LinearEncoder, self).__init__()

		self.fc = nn.Linear(nin, nout)

	def forward(self, x):

		x = x.view(x.size(0), -1) #flatten tensor
		z = self.fc(x)

		return z


class NonLinearEncoder(nn.Module):
	def __init__(self, nin, nout):
		super(LinearEncoder, self).__init__()

		self.fc1 = nn.Linear(nin, 500)
		self.fc2 = nn.Linear(500, nout)

	def forward(self, x):

		x = x.view(x.size(0), -1) #flatten tensor
		x = F.relu(self.fc1(x))
		z = self.fc2(x)

		return z


class Decoder(nn.Module):
	def __init__(self, nin, nout):
		super(Decoder, self).__init__()

		self.fc1 = nn.Linear(nin, 500)
		self.fc2 = nn.Linear(500, 500)
		self.fc3 = nn.Linear(500, nout)

	def forward(self, z):

		z = F.relu(self.fc1(z))
		z = F.relu(self.fc2(z))
		x = self.fc3(z)

		return x


class EncoderDecoder(nn.Module):
	def __init__(self, 
				nin, 
				args):
		super(EncoderDecoder, self).__init__()
		self.loss = args.loss
		self.linear = args.linear
		self.zdim = args.zdim
		self.args = args

		enc_nout = self.zdim
	    if self.loss == 'vae' or self.loss == 'bvae':
	        enc_nout = 2*self.zdim 

		if self.linear:
			enc_net = [nn.Linear(nin, enc_nout)]
		else:
			enc_net = [nn.Linear(nin, 500),
						nn.ReLU(),
						nn.Linear(500, enc_nout)]

		self.enc_net = nn.Sequential(*enc_net)

		self.dec_net = nn.Sequential([nn.Linear(self.zdim, 500),
									nn.ReLU(),
									nn.Linear(500, 500),
									nn.ReLU(),
									nn.Linear(500, 500),
									nn.Sigmoid()])

	def encode(self, x, linear):

		x = x.view(x.size(0), -1) #flatten tensor
		z_param = self.enc_net(x)

		return z_param

	def decoder(self, z):

		x = self.dec_net(z)

		return x

	def forward(self, x):

		loss = self.loss

		if loss == 'dae':
			x
		z_param = self.encode(x)

		if loss == 'uae':

		elif loss == 'ae'
		elif loss == 'vae' or loss == 'bvae':

		elif loss == 'uae'













