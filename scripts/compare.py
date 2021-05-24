import argparse
import os
import torch
import pandas as pd
import numpy
from matplotlib import pyplot

from attrdict import AttrDict

from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)


def theta_loss(pred_traj_fake,pred_traj_gt,last_obs_traj,loss_mask,consider_ped=None,mode='average'):
    pred_len = pred_traj_fake.size(0)
    batch = pred_traj_fake.size(1)

    delta_fake = (pred_traj_fake[0,:,:]-last_obs_traj).view(1,batch,2)
    delta_gt = (pred_traj_gt[0,:,:]-last_obs_traj).view(1,batch,2)
    for step in range(1,pred_len):
        delta_fake = torch.cat([delta_fake,(pred_traj_fake[step,:,:]-pred_traj_fake[step-1,:,:]).view(1,batch,2)],dim = 0)
        delta_gt = torch.cat([delta_gt,(pred_traj_gt[step,:,:]-pred_traj_gt[step-1,:,:]).view(1,batch,2)],dim = 0)
          
    theta_fake = torch.atan2(delta_fake[:,:,1],delta_fake[:,:,0])
    theta_gt = torch.atan2(delta_gt[:,:,1],delta_gt[:,:,0])
    loss = torch.sum(torch.abs(theta_gt - theta_fake),dim=0)
    if consider_ped is not None:
        loss = loss*consider_ped
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == 'final':
        print(loss[-1].size,loss[-1])
        return loss[-1]
    elif mode == 'raw':
        return loss


def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]
    print("paths", paths)
    color='r'
    for path in paths:
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        print(checkpoint.keys())
        G_total_loss = checkpoint['metrics_train']['fte']
        print(len(G_total_loss))
        x = numpy.arange(len(G_total_loss))
        y = G_total_loss

        fig = pyplot.figure()
        ax = fig.add_subplot(111)
        # ax.set_ylim(0,10)
        pyplot.plot(x,y, color=color, label=path)
        color='g'
        # for i,j in zip(x,y):
        #     ax.annotate(str(j),xy=(i,j))

    pyplot.show()



if __name__ == '__main__':
    # pred_traj_fake = torch.rand(8,400,2)
    # pred_traj_gt = torch.rand(8,400,2)
    # last_obs_traj = torch.rand(400,2)
    # loss_mask = torch.rand(1,400,2)
    # theta_loss(pred_traj_fake,pred_traj_gt,last_obs_traj,loss_mask,mode='final')
    y = torch.tensor([1,2,3])
    x = torch.tensor([1,4,6])
    print(y)
    print(torch.atan2(y,x),"theta degree")

    args = parser.parse_args()
    main(args)



