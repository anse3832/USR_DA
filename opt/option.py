import argparse


parser = argparse.ArgumentParser(description='BebyGAN')

# Hardware specifications
parser.add_argument('--gpu_id', type=str, help='specify GPU ID to use')
parser.add_argument('--num_workers', type=int, default=8)

# Data specifications
parser.add_argument('--dir_data', type=str, default='/mnt/Dataset/anse_data/USRdata/NTIRE_RWSR', help='dataset root directory')
parser.add_argument('--scale', type=int, default=4, help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=64, help='LR patch size') # default = 128 (in the paper)

# Train specifications
parser.add_argument('--epochs', type=int, default=35000, help='total epochs')
parser.add_argument('--batch_size', type=int, default=4, help='size of each batch') # default = 8 (in the paper)

# Optimizer specificaions
parser.add_argument('--lr_G', type=float, default=1e-4, help='initial learning rate of generator')
parser.add_argument('--lr_D', type=float, default=1e-4, help='initial learning rate of discriminator')
parser.add_argument('--beta1', type=float, default=0.9, help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.99, help='ADAM beta2')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')

# Scheduler specifications
parser.add_argument('--interval1', type=int, default=2.5e5, help='1st step size (iteration)')
parser.add_argument('--interval2', type=int, default=3.5e5, help='2nd step size (iteration)')
parser.add_argument('--interval3', type=int, default=4.5e5, help='3rd step size (iteration)')
parser.add_argument('--interval4', type=int, default=5.5e5, help='4th step size (iteration)')
parser.add_argument('--gamma_G', type=float, default=0.5, help='generator learning rate decay ratio')
parser.add_argument('--gamma_D', type=float, default=0.5, help='discriminator learning rate decay ratio')

# Train specificaions
parser.add_argument('--snap_path', type=str, default='./weights', help='path to save model weights')
parser.add_argument('--save_freq', type=str, default=50, help='save model frequency (epoch)')

# checkpoint
parser.add_argument('--checkpoint', type=str, default=None, help='load checkpoint')

# Optimizer specifications 
parser.add_argument('--lambda_align', type=float, default=0.01, help='L1 loss weight')
parser.add_argument('--lambda_rec', type=float, default=1.0, help='back-projection loss weight')
parser.add_argument('--lambda_res', type=float, default=1.0, help='perceptual loss weight')
parser.add_argument('--lambda_sty', type=float, default=0.01, help='style loss weight')
parser.add_argument('--lambda_idt', type=float, default=0.01, help='identity loss weight')
parser.add_argument('--lambda_cyc', type=float, default=1, help='cycle loss weight')

parser.add_argument('--lambda_percept', type=float, default=0.01, help='perceptual loss weight')
parser.add_argument('--lambda_adv', type=float, default=0.01, help='adversarial loss weight')

# generator & discriminator specifications
parser.add_argument('--n_disc', type=int, default=1, help='number of iteration for discriminator update in one epoch')
parser.add_argument('--n_gen', type=int, default=2, help='number of iteration for generator update in one epoch')

# encoder & decoder specifications
parser.add_argument('--n_hidden_feats', type=int, default=64, help='number of feature vectors in hidden layer')
parser.add_argument('--n_sr_feats', type=int, default=64, help='number of feature vectors in RRDB layer')

# test specifications
parser.add_argument('--weights', type=str, help='load weights for test')
parser.add_argument('--dir_test', type=str, help='directory of test images')
parser.add_argument('--results', type=str, default='./results/', help='directory of test results')

args = parser.parse_args()