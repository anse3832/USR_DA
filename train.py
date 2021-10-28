import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from opt.option import args
from data.LQGT_dataset import LQGTDataset
from util.utils import RandCrop, RandHorizontalFlip, RandRotate, ToTensor, VGG19PerceptualLoss
from model import encoder, decoder, discriminator

from tqdm import tqdm


# device setting
if args.gpu_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    print('using GPU %s' % args.gpu_id)
else:
    print('use --gpu_id to specify GPU ID to use')
    exit()


# make directory for saving weights
if not os.path.exists(args.snap_path):
    os.mkdir(args.snap_path)


# load training dataset
train_dataset = LQGTDataset(
    db_path=args.dir_data,
    transform=transforms.Compose([RandCrop(args.patch_size, args.scale), RandHorizontalFlip(), RandRotate(), ToTensor()])
)
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    drop_last=True,
    shuffle=True
)


# define model (generator)
model_Enc = encoder.Encoder_RRDB(num_feat=args.n_hidden_feats).cuda()
model_Dec_Id = decoder.Decoder_Id_RRDB(num_in_ch=args.n_hidden_feats).cuda()
model_Dec_SR = decoder.Decoder_SR_RRDB(num_in_ch=args.n_hidden_feats).cuda()

# define model (discriminator)
model_Disc_feat = discriminator.DiscriminatorVGG(in_ch=args.n_hidden_feats, image_size=args.patch_size).cuda()
model_Disc_img_LR = discriminator.DiscriminatorVGG(in_ch=3, image_size=args.patch_size).cuda()
model_Disc_img_HR = discriminator.DiscriminatorVGG(in_ch=3, image_size=args.scale*args.patch_size).cuda()
# model_Disc_feat = discriminator.UNetDiscriminator(num_in_ch=64).cuda()
# model_Disc_img_LR = discriminator.UNetDiscriminator(num_in_ch=3).cuda()
# model_Disc_img_HR = discriminator.UNetDiscriminator(num_in_ch=3).cuda()


# loss
loss_L1 = nn.L1Loss().cuda()
loss_MSE = nn.MSELoss().cuda()
loss_adversarial = nn.BCEWithLogitsLoss().cuda()
loss_percept = VGG19PerceptualLoss().cuda()


# optimizer 
params_G = list(model_Enc.parameters()) + list(model_Dec_Id.parameters()) + list(model_Dec_SR.parameters())
optimizer_G = optim.Adam(
    params_G,
    lr=args.lr_G,
    betas=(args.beta1, args.beta2),
    weight_decay=args.weight_decay,
    amsgrad=True
)
params_D = list(model_Disc_feat.parameters()) + list(model_Disc_img_LR.parameters()) + list(model_Disc_img_HR.parameters())
optimizer_D = optim.Adam(
    params_D,
    lr=args.lr_D,
    betas=(args.beta1, args.beta2),
    weight_decay=args.weight_decay,
    amsgrad=True
)

# Scheduler
iter_indices = [args.interval1, args.interval2, args.interval3]
scheduler_G = optim.lr_scheduler.MultiStepLR(
    optimizer=optimizer_G,
    milestones=iter_indices,
    gamma=0.5
)
scheduler_D = optim.lr_scheduler.MultiStepLR(
    optimizer=optimizer_D,
    milestones=iter_indices,
    gamma=0.5
)


# load model weights & optimzer % scheduler
if args.checkpoint is not None:
    checkpoint = torch.load(args.checkpoint)

    model_Enc.load_state_dict(checkpoint['model_Enc'])
    model_Dec_Id.load_state_dict(checkpoint['model_Dec_Id'])
    model_Dec_SR.load_state_dict(checkpoint['model_Dec_SR'])
    model_Disc_feat.load_state_dict(checkpoint['model_Disc_feat'])
    model_Disc_img_LR.load_state_dict(checkpoint['model_Disc_img_LR'])
    model_Disc_img_HR.load_state_dict(checkpoint['model_Disc_img_HR'])

    optimizer_D.load_state_dict(checkpoint['optimizer_D'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G'])

    scheduler_D.load_state_dict(checkpoint['scheduler_D'])
    scheduler_G.load_state_dict(checkpoint['scheduler_G'])

    start_epoch = checkpoint['epoch']
else:
    start_epoch = 0


# training
for epoch in range(start_epoch, args.epochs):
    # generator
    model_Enc.train()
    model_Dec_Id.train()
    model_Dec_SR.train()

    # discriminator
    model_Disc_feat.train()
    model_Disc_img_LR.train()
    model_Disc_img_HR.train()
    
    running_loss_D_total = 0.0
    running_loss_G_total = 0.0

    running_loss_align = 0.0
    running_loss_rec = 0.0
    running_loss_res = 0.0
    running_loss_sty = 0.0
    running_loss_idt = 0.0
    running_loss_cyc = 0.0

    iter = 0    

    for data in tqdm(train_loader):
        iter += 1

        ########################
        #       data load      #
        ########################
        X_t, Y_s = data['img_LQ'], data['img_GT']

        ds4 = nn.Upsample(scale_factor=1/args.scale, mode='bicubic')
        X_s = ds4(Y_s)

        X_t = X_t.cuda(non_blocking=True)
        X_s = X_s.cuda(non_blocking=True)
        Y_s = Y_s.cuda(non_blocking=True)

        # real label and fake label
        batch_size = X_t.size(0)
        real_label = torch.full((batch_size, 1), 1, dtype=X_t.dtype).cuda(non_blocking=True)
        fake_label = torch.full((batch_size, 1), 0, dtype=X_t.dtype).cuda(non_blocking=True)


        ########################
        # (1) Update D network #
        ########################
        model_Disc_feat.zero_grad()
        model_Disc_img_LR.zero_grad()
        model_Disc_img_HR.zero_grad()

        for i in range(args.n_disc):
            # generator output (feature domain)
            F_t = model_Enc(X_t)
            F_s = model_Enc(X_s)

            # 1. feature aligment loss (discriminator)
            # output of discriminator (feature domain) (b x c(=1) x h x w)
            output_Disc_F_t = model_Disc_feat(F_t.detach())
            output_Disc_F_s = model_Disc_feat(F_s.detach())
            # discriminator loss (feature domain)
            loss_Disc_F_t = loss_MSE(output_Disc_F_t, fake_label)
            loss_Disc_F_s = loss_MSE(output_Disc_F_s, real_label)
            loss_Disc_feat_align = (loss_Disc_F_t + loss_Disc_F_s) / 2

            # 2. SR reconstruction loss (discriminator)
            # generator output (image domain)
            Y_s_s = model_Dec_SR(F_s)
            # output of discriminator (image domain)
            output_Disc_Y_s_s = model_Disc_img_HR(Y_s_s.detach())
            output_Disc_Y_s = model_Disc_img_HR(Y_s)
            # discriminator loss (image domain)
            loss_Disc_Y_s_s = loss_MSE(output_Disc_Y_s_s, fake_label)
            loss_Disc_Y_s = loss_MSE(output_Disc_Y_s, real_label)
            loss_Disc_img_rec = (loss_Disc_Y_s_s + loss_Disc_Y_s) / 2

            # 4. Target degradation style loss
            # generator output (image domain)
            X_s_t = model_Dec_Id(F_s)
            # output of discriminator (image domain)
            output_Disc_X_s_t = model_Disc_img_LR(X_s_t.detach())
            output_Disc_X_t = model_Disc_img_LR(X_t)
            # discriminator loss (image domain)
            loss_Disc_X_s_t = loss_MSE(output_Disc_X_s_t, fake_label)
            loss_Disc_X_t = loss_MSE(output_Disc_X_t, real_label)
            loss_Disc_img_sty = (loss_Disc_X_s_t + loss_Disc_X_t) / 2

            # 6. Cycle loss
            # generator output (image domain)
            Y_s_t_s = model_Dec_SR(model_Enc(model_Dec_Id(F_s)))
            # output of discriminator (image domain)
            output_Disc_Y_s_t_s = model_Disc_img_HR(Y_s_t_s.detach())
            output_Disc_Y_s = model_Disc_img_HR(Y_s)
            # discriminator loss (image domain)
            loss_Disc_Y_s_t_s = loss_MSE(output_Disc_Y_s_t_s, fake_label)
            loss_Disc_Y_s = loss_MSE(output_Disc_Y_s, real_label)
            loss_Disc_img_cyc = (loss_Disc_Y_s_t_s + loss_Disc_Y_s) / 2

            # discriminator weight update
            loss_D_total = loss_Disc_feat_align + loss_Disc_img_rec + loss_Disc_img_sty + loss_Disc_img_cyc
            loss_D_total.backward()
            optimizer_D.step()
        scheduler_D.step()


        ########################
        # (2) Update G network #
        ########################
        model_Enc.zero_grad()
        model_Dec_Id.zero_grad()
        model_Dec_SR.zero_grad()

        for i in range(args.n_gen):
            # generator output (feature domain)
            F_t = model_Enc(X_t)
            F_s = model_Enc(X_s)

            # 1. feature alignment loss (generator)
            # output of discriminator (feature domain)
            output_Disc_F_t = model_Disc_feat(F_t)
            output_Disc_F_s = model_Disc_feat(F_s)
            # generator loss (feature domain)
            loss_G_F_t = loss_MSE(output_Disc_F_t, (real_label + fake_label)/2)
            loss_G_F_s = loss_MSE(output_Disc_F_s, (real_label + fake_label)/2)
            L_align_E = loss_G_F_t + loss_G_F_s

            # 2. SR reconstruction loss
            # generator output (image domain)
            Y_s_s = model_Dec_SR(F_s)
            # output of discriminator (image domain)
            output_Disc_Y_s_s = model_Disc_img_HR(Y_s_s)
            # L1 loss
            loss_L1_rec = loss_L1(Y_s.detach(), Y_s_s)
            # perceptual loss
            loss_percept_rec = loss_percept(Y_s.detach(), Y_s_s)
            # adversatial loss
            loss_G_Y_s_s = loss_MSE(output_Disc_Y_s_s, real_label)
            L_rec_G_SR = loss_L1_rec + args.lambda_percept*loss_percept_rec + args.lambda_adv*loss_G_Y_s_s

            # 3. Target LR restoration loss
            X_t_t = model_Dec_Id(F_t)
            L_res_G_t = loss_L1(X_t, X_t_t)

            # 4. Target degredation style loss
            # generator output (image domain)
            X_s_t = model_Dec_Id(F_s)
            # output of discriminator (img domain)
            output_Disc_X_s_t = model_Disc_img_LR(X_s_t)
            # generator loss (feature domain)
            loss_G_X_s_t = loss_MSE(output_Disc_X_s_t, real_label)
            L_sty_G_t = loss_G_X_s_t

            # 5. Feature identity loss
            F_s_tilda = model_Enc(model_Dec_Id(F_s))
            L_idt_G_t = loss_L1(F_s, F_s_tilda)

            # 6. Cycle loss
            # generator output (image domain)
            Y_s_t_s = model_Dec_SR(model_Enc(model_Dec_Id(F_s)))
            # output of discriminator (image domain)
            output_Disc_Y_s_t_s = model_Disc_img_HR(Y_s_t_s)
            # L1 loss
            loss_L1_cyc = loss_L1(Y_s.detach(), Y_s_t_s)
            # perceptual loss
            loss_percept_cyc = loss_percept(Y_s.detach(), Y_s_t_s)
            # adversarial loss 
            loss_Y_s_t_s = loss_MSE(output_Disc_Y_s_t_s, real_label)
            L_cyc_G_t_G_SR = loss_L1_cyc + args.lambda_percept*loss_percept_cyc + args.lambda_adv*loss_Y_s_t_s

            # generator weight update
            loss_G_total = args.lambda_align*L_align_E + args.lambda_rec*L_rec_G_SR + args.lambda_res*L_res_G_t + args.lambda_sty*L_sty_G_t + args.lambda_idt*L_idt_G_t + args.lambda_cyc*L_cyc_G_t_G_SR
            loss_G_total.backward()
            optimizer_G.step()
        scheduler_G.step()


        ########################
        #     compute loss     #
        ########################
        running_loss_D_total += loss_D_total.item()
        running_loss_G_total += loss_G_total.item()

        running_loss_align += L_align_E.item()
        running_loss_rec += L_rec_G_SR.item()
        running_loss_res += L_res_G_t.item()
        running_loss_sty += L_sty_G_t.item()
        running_loss_idt += L_idt_G_t.item()
        running_loss_cyc += L_cyc_G_t_G_SR.item()

    print('epoch:%d, lr:%f, loss_D_total:%f, loss_G_total:%f, loss_align:%f, loss_rec:%f, loss_res:%f, loss_sty:%f, loss_idt:%f, loss_cyc:%f' % (epoch, optimizer_G.param_groups[0]['lr'], running_loss_D_total/iter, running_loss_G_total/iter, running_loss_align/iter, running_loss_rec/iter, running_loss_res/iter, running_loss_sty/iter, running_loss_idt/iter, running_loss_cyc/iter))


    if (epoch+1) % args.save_freq == 0:
        weights_file_name = 'epoch_%d.pth' % (epoch+1)
        weights_file = os.path.join(args.snap_path, weights_file_name)
        torch.save({
            'epoch': epoch,

            'model_Enc': model_Enc.state_dict(),
            'model_Dec_Id': model_Dec_Id.state_dict(),
            'model_Dec_SR': model_Dec_SR.state_dict(),
            'model_Disc_feat': model_Disc_feat.state_dict(),
            'model_Disc_img_LR': model_Disc_img_LR.state_dict(),
            'model_Disc_img_HR': model_Disc_img_HR.state_dict(),

            'optimizer_D': optimizer_D.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),

            'scheduler_D': scheduler_D.state_dict(),
            'scheduler_G': scheduler_G.state_dict(),
        }, weights_file)
        print('save weights of epoch %d' % (epoch+1))