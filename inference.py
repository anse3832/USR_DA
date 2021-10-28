import os
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torchvision

from model import encoder, decoder
from opt.option import args


# device setting
if args.gpu_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    print('using GPU %s' % args.gpu_id)
else:
    print('use --gpu_id to specify GPU ID to use')
    exit()


# make directory for saving weights
if not os.path.exists(args.results):
    os.mkdir(args.results)


# numpy array -> torch tensor
class ToTensor(object):
    def __call__(self, sample):
        sample = np.transpose(sample, (2, 0, 1))
        sample = torch.from_numpy(sample)
        return sample


# create model
# model_Enc = encoder.Encoder().cuda()
# model_Dec_SR = decoder.Decoder_SR().cuda()
model_Enc = encoder.Encoder_RRDB(num_feat=args.n_hidden_feats).cuda()
model_Dec_SR = decoder.Decoder_SR_RRDB(num_in_ch=args.n_hidden_feats).cuda()


# load weights
checkpoint = torch.load(args.weights)
model_Enc.load_state_dict(checkpoint['model_Enc'])
model_Dec_SR.load_state_dict(checkpoint['model_Dec_SR'])
model_Enc.eval()
model_Dec_SR.eval()

# input transform
transforms = torchvision.transforms.Compose([ToTensor()])


filenames = os.listdir(args.dir_test)
filenames.sort()
for filename in tqdm(filenames):
    img_name = os.path.join(args.dir_test, filename)
    ext = os.path.splitext(img_name)[-1]
    if ext =='.png':
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img).astype('float32') / 255
        # img = img[0:256, 0:256, :]
        
        img = transforms(img)
        img = torch.tensor(img.cuda()).unsqueeze(0)

        # inference output
        feat = model_Enc(img)
        out = model_Dec_SR(feat)

        min_max = (0, 1)
        out = out.detach()[0].float().cpu()

        out = out.squeeze().float().cpu().clamp_(*min_max)
        out = (out - min_max[0]) / (min_max[1] - min_max[0])
        out = out.numpy()
        out = np.transpose(out[[2, 1, 0], :, :], (1, 2, 0))

        out = (out*255.0).round()
        out = out.astype(np.uint8)

        # result image save (b x c x h x w (torch tensor) -> h x w x c (numpy array))
        # out = out.data.cpu().squeeze().numpy()
        # out = np.clip(out, 0, 1)
        # out = np.transpose(out, (1, 2, 0))

        cv2.imwrite('%s_out.png' %(os.path.join(args.results, filename)[:-4]), out)





