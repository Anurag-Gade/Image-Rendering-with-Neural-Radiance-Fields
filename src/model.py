from utils import *
from encoder_params import *
from train_params import *

import_libraries()
sine_activation, positional_encoding = encoder_par()
epochs, alpha, batch_size, momentum, up_scale, device = train_par()

class NERF_Model(nn.Module):
  def __init__(self,layers,units,encoding_degree,sine_activation,positional_encoding):
    super(NERF_Model,self).__init__()
    self.layers = layers
    self.units = units
    self.encoding_degree = encoding_degree
    self.sine_activation = sine_activation
    self.positional_encoding = positional_encoding
    if(self.positional_encoding == False):
      self.input = nn.Linear(2,self.units)
    else:
      self.input = nn.Linear(4*self.encoding_degree, self.units)
    
    linear_list = []
    for _ in range(self.layers):
      linear_list.append(nn.Linear(self.units, self.units))
    
    self.layers = nn.ModuleList(linear_list)
    '''
    The number of units in the output of our implicit neural network is supposed to be 3, as our output requires to be in the RGB format
    '''
    self.output = nn.Linear(self.units,3)   

    self.activation = nn.ReLU(inplace=True) 
    self.output_activation = nn.Softmax()

  def forward(self,t):
    if(self.positional_encoding == False):
      u = self.input(t)
    else:
      sinusoidal_list = []
      for i in range(self.encoding_degree):
        sinusoidal_list += [torch.sin(2**i*np.pi*t[:,0].unsqueeze(1)),torch.cos(2**i*np.pi*t[:,0].unsqueeze(1))]
        sinusoidal_list += [torch.sin(2**i*np.pi*t[:,1].unsqueeze(1)),torch.cos(2**i*np.pi*t[:,1].unsqueeze(1))]

      embedded = torch.concat(sinusoidal_list, dim=-1)
      u = self.input(embedded)

    for layer in self.layers:
      if self.sine_activation:
        u = torch.sin(layer(u))
      
      else:
        u = self.activation(layer(u))

    return self.output(u)
