import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from models import Generator
from datasets import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--outf', default='output', help='checkpoint file path')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()

# Load state dicts
generator_A2B = os.path.join(opt.outf, 'netG_A2B.pth')
generator_B2A = os.path.join(opt.outf, 'netG_B2A.pth')
netG_A2B.load_state_dict(torch.load(generator_A2B))
netG_B2A.load_state_dict(torch.load(generator_B2A))

# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

# Dataset loader
transforms_ = [ transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='test'),
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
###################################

###### Testing######
outf_ABA = os.path.join(opt.outf, 'ABA')
outf_BAB = os.path.join(opt.outf, 'BAB')
# Create output dirs if they don't exist
if not os.path.exists(outf_ABA):
    os.makedirs(outf_ABA)
if not os.path.exists(outf_BAB):
    os.makedirs(outf_BAB)

for i, batch in enumerate(dataloader):
    # Set model input
    real_A = Variable(input_A.copy_(batch['A']))
    real_B = Variable(input_B.copy_(batch['B']))
    real_A_ = 0.5 * (real_A + 1.0)
    real_B_ = 0.5 * (real_B + 1.0)

    # Generate output
    fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
    fake_A = 0.5*(netG_B2A(real_B).data + 1.0)

    # Recovered output
    recon_A = 0.5*(netG_B2A(fake_B).data + 1.0)
    recon_B = 0.5*(netG_A2B(fake_A).data + 1.0)

    # Save image files
    save_image(real_A_, os.path.join(outf_ABA, 'A_%04d.png' % (i+1)))
    save_image(fake_B, os.path.join(outf_ABA, 'AB_%04d.png' % (i+1)))
    save_image(recon_A, os.path.join(outf_ABA, 'ABA_%04d.png' % (i+1)))
    save_image(real_B_, os.path.join(outf_BAB, 'B_%04d.png' % (i+1)))
    save_image(fake_A, os.path.join(outf_BAB, 'BA_%04d.png' % (i+1)))
    save_image(recon_B, os.path.join(outf_BAB, 'BAB_%04d.png' % (i+1)))

    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

sys.stdout.write('\n')
###################################
