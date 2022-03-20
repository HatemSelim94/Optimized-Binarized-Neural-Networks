import matplotlib.pyplot as plt
from numpy import squeeze
import seaborn as sb
import torch
import os
import random as rand
def plot_tensor_dist(input, save= False, show=False, file_name = None, density = False):
    file_dir = file_name.split('/')[:-1]
    file_dir = '/'.join(file_dir)
    if density:
      file_name = file_name.split('/')[-1]
      file_dir = os.path.join(file_dir,'density')
      file_name = file_dir+'/'+file_name
    if not os.path.exists(file_dir):
      os.makedirs(file_dir)
    if file_name is None:
      file_name = 'Dist'
    with torch.no_grad():
      x = input[0,:,:,:].view(-1).numpy()
    min = x.min().round()
    max = x.max().round()
    fig, axs = plt.subplots(1, 1,
                        tight_layout = True)
    
    #axs.hist(x, bins = 100, color='g',density=density)
    sb.displot(x = x  , kind = 'kde' , color = 'green', fill=True)
    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    axs.spines['left'].set_position('zero')
    #axs.spines['bottom'].set_position('center')
	# Eliminate upper and right axes
    axs.spines['right'].set_color('none')
    axs.spines['top'].set_color('none')
    plt.xticks([-max, -min,0,min,max])
    plt.yticks([])
    # Show plot
    if save:
        plt.savefig(file_name,dpi=240)
    if show:
        plt.show()
    plt.cla()
    plt.close('all')
    plt.rcParams.update({'figure.max_open_warning': 0})


def plot_weight(weight, save=False, show=False, file_name=None, random=False):
  _, c, _,_ = weight.shape
  if random:
    nums = rand.sample(range(c), 3)
    for num in nums:
      feature_map = weight[0,num, :,:]
      plt.imshow(feature_map, cmap='gray')
      if save:
        plt.savefig(file_name+f'weight{num}',dpi=240)
      if show:
        plt.show()
      plt.cla()
      plt.close('all')
      plt.rcParams.update({'figure.max_open_warning': 0})
  else:
    feature_map = weight[0,0, :,:].squeeze().squeeze()
    plt.imshow(feature_map, cmap='gray')
    if save:
      plt.savefig(file_name+f'weight{0}',dpi=240)
    if show:
      plt.show()
    plt.cla()
    plt.close('all')
    plt.rcParams.update({'figure.max_open_warning': 0})
