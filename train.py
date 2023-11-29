import os
import time
import random
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.backends import cudnn

from util import util
from util.plot import plot_batch

from models.projected_model import fsModel
from data.data_loader_Swapping import GetLoader
from models.vgg_feature import VGGFeature
from tqdm import tqdm
from models.mouth_net import MouthNet
from models import pytorch_ssim
from models.back_model import BiSeNet

def str2bool(v):
    return v.lower() in ('true')

def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w*h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G

def calc_CD_loss(model,tea_outs,stu_outs):
    losses = []
    for (netA, Tact, Sact) in zip(model.netAs, tea_outs, stu_outs):
        Sact_A = netA(Sact)
        source, target = Sact_A, Tact.detach()
        source = source.mean(dim=(2, 3), keepdim=False)
        target = target.mean(dim=(2, 3), keepdim=False)
        loss = torch.mean(torch.pow(source - target, 2))
        losses.append(loss)
    return sum(losses)

def encode_segmentation(image,parsing):
    num_of_class = 17
    back_map = torch.zeros_like(parsing)
    for pi in range(1, num_of_class + 1):
        if pi not in [0,7,8,9,14,15,16,17,18]:
            valid_index = torch.where(parsing == pi)
            back_map[valid_index] = 1
    back_map = back_map.unsqueeze(1).repeat(1,3,1,1)
    result = image * (1 - back_map)
    return result

class TrainOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        
    def initialize(self):
        self.parser.add_argument('--name', type=str, default='simswap', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', default='0')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--isTrain', type=str2bool, default='True')

        # input/output sizes       
        self.parser.add_argument('--batchSize', type=int, default=4, help='input batch size')       

        # for displays
        self.parser.add_argument('--use_tensorboard', type=str2bool, default='False')

        # for training
        self.parser.add_argument('--dataset', type=str, default="/path/to/VGGFace2", help='path to the face swapping dataset')
        self.parser.add_argument('--continue_train', type=str2bool, default='False', help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='./checkpoints/simswap224_test', help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='10000', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=10000, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=10000, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--Gdeep', type=str2bool, default='False')
        self.parser.add_argument('--use_back', type=bool, default=True, help='weight for mouth loss')

        # for discriminators         
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        self.parser.add_argument('--lambda_id', type=float, default=30.0, help='weight for id loss')
        self.parser.add_argument('--lambda_rec', type=float, default=10.0, help='weight for reconstruction loss')
        self.parser.add_argument('--lambda_CD', type=float, default=5e1, help='weight for CD loss')
        self.parser.add_argument('--lambda_SSIM', type=float, default=1e1, help='weight for SSIM loss')
        self.parser.add_argument('--lambda_style', type=float, default=1e4, help='weight for style loss')
        self.parser.add_argument('--lambda_feature', type=float, default=1e1, help='weight for feature loss')
        self.parser.add_argument('--lambda_tv', type=float, default=1e-5, help='weight for total v loss')
        self.parser.add_argument('--lambda_mouth', type=float, default=0.1, help='weight for mouth loss')
        self.parser.add_argument('--lambda_disc', type=float, default=1.0, help='weight for mouth loss')
        self.parser.add_argument('--lambda_back', type=float, default=1, help='weight for mouth loss')

        self.parser.add_argument("--Arc_path", type=str, default='arcface_model/arcface_checkpoint.tar', help="run ONNX model via TRT")
        self.parser.add_argument("--Parser_path", type=str, default='weights/79999_iter.pth', help="run ONNX model via TRT")
        self.parser.add_argument("--Mouth_path", type=str, default='weights/mouth_net_28_56_84_112.pth', help="run ONNX model via TRT")

        self.parser.add_argument("--total_step", type=int, default=1000000, help='total training step')
        self.parser.add_argument("--log_frep", type=int, default=200, help='frequence for printing log information')
        self.parser.add_argument("--sample_freq", type=int, default=200, help='frequence for sampling')
        self.parser.add_argument("--model_freq", type=int, default=10000, help='frequence for saving the model')

        self.isTrain = True
        
    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        if self.opt.isTrain:
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            util.mkdirs(expr_dir)
            if save and not self.opt.continue_train:
                file_name = os.path.join(expr_dir, 'opt.txt')
                with open(file_name, 'wt') as opt_file:
                    opt_file.write('------------ Options -------------\n')
                    for k, v in sorted(args.items()):
                        opt_file.write('%s: %s\n' % (str(k), str(v)))
                    opt_file.write('-------------- End ----------------\n')
        return self.opt


if __name__ == '__main__':

    opt         = TrainOptions().parse()
    iter_path   = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

    sample_path = os.path.join(opt.checkpoints_dir, opt.name, 'samples')

    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    
    log_path = os.path.join(opt.checkpoints_dir, opt.name, 'summary')

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
    else:    
        start_epoch, epoch_iter = 1, 0

    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_ids)
    print("GPU used : ", str(opt.gpu_ids))

    cudnn.benchmark = True

    model = fsModel()
    model.initialize(opt)

    n_classes = 19
    parsing_net = BiSeNet(n_classes=n_classes)
    parsing_net.cuda()
    parsing_net.load_state_dict(torch.load(opt.Parser_path))
    parsing_net.eval()
    vgg = VGGFeature().to('cuda')
    mouth_feat_dim = 128
    mouth_crop_param = (28, 56, 84, 112)
    mouth_net = MouthNet(
        bisenet=None,
        feature_dim=mouth_feat_dim,
        crop_param=mouth_crop_param
    )
    mouth_net.load_backbone(opt.Mouth_path)
    mouth_net.cuda()
    mouth_net.eval()
    mouth_net.requires_grad_(False)
    w1, h1, w2, h2 = mouth_crop_param
    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
    with open(log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ Training Loss (%s) ================\n' % now)
    optimizer_G, optimizer_D,optimizer_G_student = model.optimizer_G, model.optimizer_D,model.optimizer_G_student
    loss_avg        = 0
    refresh_count   = 0
    imagenet_std    = torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1)
    imagenet_mean   = torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1)
    train_loader    = GetLoader(opt.dataset,opt.batchSize,8,1234)
    randindex = [i for i in range(opt.batchSize)]
    random.shuffle(randindex)
    if not opt.continue_train:
        start   = 1
    else:
        start   = int(opt.which_epoch)
    total_step  = opt.total_step
    import datetime
    print("Start to train at %s"%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    model.netD.feature_network.requires_grad_(False)
    pbar = range(start, total_step)
    pbar = tqdm(pbar, total=total_step)
    loss_G = torch.Tensor([0.0])
    loss_G_ID = torch.Tensor([0.0])
    loss_G_Rec = torch.Tensor([0.0])
    loss_G_s_f = torch.Tensor([0.0])
    loss_G_Mouth = torch.Tensor([0.0])
    # Training Cycle
    for step in pbar:
        model.netG.train()
        model.netG_student.train()
        for interval in range(2):
            random.shuffle(randindex)
            src_image1, src_image2  = train_loader.next()

            if step%2 == 0:
                img_id = src_image2
            else:
                img_id = src_image2[randindex]
            with torch.no_grad():
                img_id_112 = F.interpolate(img_id,size=(112,112), mode='bicubic')
                latent_id = model.netArc(img_id_112)
                latent_id = F.normalize(latent_id, p=2, dim=1)
            if interval:
                img_fake,_        = model.netG(src_image1, latent_id)
                gen_logits,_    = model.netD(img_fake.detach(), None)
                loss_Dgen       = (F.relu(torch.ones_like(gen_logits) + gen_logits)).mean()

                real_logits,_   = model.netD(src_image2,None)
                loss_Dreal      = (F.relu(torch.ones_like(real_logits) - real_logits)).mean()

                loss_D          = loss_Dgen + loss_Dreal
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()
            else:
                img_fake, tea_outs = model.netG(src_image1, latent_id)
                # model.netD.requires_grad_(True)
                # G loss
                gen_logits,feat = model.netD(img_fake, None)

                loss_Gmain      = (-gen_logits).mean()
                img_fake_down   = F.interpolate(img_fake, size=(112,112), mode='bicubic')
                latent_fake     = model.netArc(img_fake_down)
                latent_fake     = F.normalize(latent_fake, p=2, dim=1)
                loss_G_ID = (1 - F.cosine_similarity(latent_id, latent_fake, 1)).mean()
                real_feat = model.netD.get_feature(src_image1)
                feat_match_loss = model.criterionFeat(feat["3"],real_feat["3"])
                source_mouth = mouth_net(img_id_112[:, :, h1:h2, w1:w2])  # mouth
                result_mouth = mouth_net(img_fake_down[:, :, h1:h2, w1:w2])  # mouth
                loss_G_Mouth = (1 - F.cosine_similarity(source_mouth, result_mouth, 1)).mean()
                if opt.use_back:
                    img_tgt_512 = F.interpolate(src_image1, size=512, mode="bilinear", align_corners=True)
                    parsing_tgt = parsing_net(img_tgt_512)[0].argmax(1)
                    tgt_back = encode_segmentation(img_tgt_512, parsing_tgt)
                    tgt_back = F.interpolate(tgt_back, size=256, mode="bilinear", align_corners=True)

                    img_fake_512 = F.interpolate(img_fake, size=512, mode="bilinear", align_corners=True)
                    parsing_fake = parsing_net(img_fake_512)[0].argmax(1)
                    fake_back = encode_segmentation(img_fake_512, parsing_fake)
                    fake_back = F.interpolate(fake_back, size=256, mode="bilinear", align_corners=True)

                    back_pixel_loss = F.l1_loss(tgt_back, fake_back) * opt.lambda_rec
                    tgt_back_features = vgg(tgt_back)
                    fake_back_features = vgg(fake_back)
                    tgt_back_gram = [gram(fmap) for fmap in tgt_back_features]
                    fake_back_gram = [gram(fmap) for fmap in fake_back_features]
                    loss_back_style = 0
                    for i in range(len(tgt_back_gram)):
                        loss_back_style += opt.lambda_style * F.l1_loss(tgt_back_gram[i], fake_back_gram[i])
                    tgt_back_recon, fake_back_recon = tgt_back_features[1], fake_back_features[1]
                    loss_back_feature = opt.lambda_feature * F.l1_loss(tgt_back_recon, fake_back_recon)
                    loss_G_Back = back_pixel_loss + loss_back_style + loss_back_feature
                    loss_G = loss_Gmain + loss_G_ID * opt.lambda_id + feat_match_loss * opt.lambda_feat + loss_G_Mouth * opt.lambda_mouth + loss_G_Back * opt.lambda_back
                else:
                    loss_G = loss_Gmain + loss_G_ID * opt.lambda_id + feat_match_loss * opt.lambda_feat + loss_G_Mouth * opt.lambda_mouth

                if step%2 == 0:
                    #G_Rec
                    loss_G_Rec  = model.criterionRec(img_fake, src_image1) * opt.lambda_rec
                    loss_G      += loss_G_Rec
                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()

                for stu_step in range(4):
                    Sfake, stu_outs = model.netG_student(src_image1, latent_id)
                    teacher_image = img_fake.detach()
                    loss_G_student = 0
                    ssim_loss = pytorch_ssim.SSIM()
                    loss_G_SSIM = (1 - ssim_loss(Sfake, teacher_image)) * opt.lambda_SSIM
                    Tfeatures = vgg(teacher_image)
                    Sfeatures = vgg(Sfake)

                    Tgram = [gram(fmap) for fmap in Tfeatures]
                    Sgram = [gram(fmap) for fmap in Sfeatures]
                    loss_G_style = 0
                    for i in range(len(Tgram)):
                        loss_G_style += opt.lambda_style * F.l1_loss(Sgram[i], Tgram[i])

                    Srecon, Trecon = Sfeatures[1], Tfeatures[1]
                    loss_G_feature = opt.lambda_feature * F.l1_loss(Srecon, Trecon)

                    diff_i = torch.sum(torch.abs(Sfake[:, :, :, 1:] - Sfake[:, :, :, :-1]))
                    diff_j = torch.sum(torch.abs(Sfake[:, :, 1:, :] - Sfake[:, :, :-1, :]))
                    loss_G_tv = opt.lambda_tv * (diff_i + diff_j)

                    loss_G_student += loss_G_SSIM + loss_G_style + loss_G_feature + loss_G_tv
                    if opt.lambda_CD:
                        loss_G_CD = calc_CD_loss(model, tea_outs, stu_outs) * opt.lambda_CD
                        loss_G_student += loss_G_CD
                    optimizer_G_student.zero_grad()
                    loss_G_student.backward()
                    optimizer_G_student.step()
                    ############## Display results and errors ##########
                    pbar.set_description(
                        (
                            f"gt:{loss_G.item():.2f};i:{loss_G_ID.item():.2f};r:{loss_G_Rec.item():.2f};m:{loss_G_Mouth.item():.2f};b:{loss_G_Back.item():.2f};gs:{loss_G_student.item():.2f};ssim: {loss_G_SSIM.item():.2f};s: {loss_G_style.item():.4f};f: {loss_G_feature.item():.4f};tv:{loss_G_tv.item():.4f}; cd:{loss_G_CD.item():.2f};")
                    )
        ### print out errors
        # Print out log info
        if (step + 1) % opt.log_frep == 0:
            # errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
            errors = {
                "G_Loss":loss_Gmain.item(),
                "G_ID":loss_G_ID.item(),
                "G_Rec":loss_G_Rec.item(),
                "G_feat_match":feat_match_loss.item(),
                "D_fake":loss_Dgen.item(),
                "D_real":loss_Dreal.item(),
                "D_loss":loss_D.item()
            }
            message = '( step: %d, ) ' % (step)
            for k, v in errors.items():
                message += '%s: %.3f ' % (k, v)
            print(message)
            with open(log_name, "a") as log_file:
                log_file.write('%s\n' % message)
        ### display output images
        if (step + 1) % opt.sample_freq == 0:
            model.netG.eval()
            model.netG_student.eval()
            with torch.no_grad():
                imgs_tea        = list()
                imgs_stu = list()
                zero_img    = (torch.zeros_like(src_image1[0,...]))
                imgs_tea.append(zero_img.cpu().numpy())
                imgs_stu.append(zero_img.cpu().numpy())
                save_img_tgt = ((src_image1.cpu()) * imagenet_std + imagenet_mean).numpy()
                save_img_sou = ((src_image2.cpu()) * imagenet_std + imagenet_mean).numpy()
                for r in range(opt.batchSize):
                    imgs_tea.append(save_img_sou[r, ...])
                    imgs_stu.append(save_img_sou[r, ...])
                arcface_112 = F.interpolate(src_image2, size=(112, 112), mode='bicubic')
                id_arc_src1 = model.netArc(arcface_112)
                id_arc_src1 = F.normalize(id_arc_src1, p=2, dim=1)
                for i in range(opt.batchSize):
                    imgs_tea.append(save_img_tgt[i, ...])
                    imgs_stu.append(save_img_tgt[i, ...])
                    image_infer = src_image1[i, ...].repeat(opt.batchSize, 1, 1, 1)
                    img_fake_tea, _ = model.netG(image_infer, id_arc_src1)
                    img_fake_stu, _ = model.netG_student(image_infer, id_arc_src1)

                    img_fake_tea = img_fake_tea.cpu()
                    img_fake_stu = img_fake_stu.cpu()

                    img_fake_tea = img_fake_tea * imagenet_std
                    img_fake_tea = img_fake_tea + imagenet_mean
                    img_fake_tea = img_fake_tea.numpy()

                    img_fake_stu = img_fake_stu * imagenet_std
                    img_fake_stu = img_fake_stu + imagenet_mean
                    img_fake_stu = img_fake_stu.numpy()
                    for j in range(opt.batchSize):
                        imgs_tea.append(img_fake_tea[j, ...])
                        imgs_stu.append(img_fake_stu[j, ...])
                print("Save test data")
                imgs_tea = np.stack(imgs_tea, axis=0).transpose(0, 2, 3, 1)
                imgs_stu = np.stack(imgs_stu, axis=0).transpose(0, 2, 3, 1)
                plot_batch(imgs_tea, os.path.join(sample_path, str(step + 1) + '_tea_.jpg'))
                plot_batch(imgs_stu, os.path.join(sample_path, str(step + 1) + '_stu_.jpg'))

        ### save latest model
        if (step+1) % opt.model_freq==0:
            print('saving the latest model (steps %d)' % (step+1))
            model.save(step+1)
            model.save_stu(step + 1)
