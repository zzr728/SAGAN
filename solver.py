from model import Generator
from model import Discriminator
from torchvision.utils import save_image
import torch.nn.functional as F
import torch
import numpy as np
import os
import time
import datetime
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix,average_precision_score
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
import cv2

class Solver(object):
    """Solver for training and testing Brainomaly."""

    def __init__(self, data_loader, config):
        """Initialize configurations."""

        # All config
        self.config = config

        # Data loader.
        self.data_loader = data_loader

        # Model configurations.
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_gp = config.lambda_gp
        self.lambda_id = config.lambda_id
        self.mode = config.mode
        self.num_workers = config.num_workers

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.device = torch.device('cuda:{}'.format(config.device) if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.save_dir = config.save_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step
        self.num_patch = 2


        # Build the model and tensorboard.
        self.build_model()

    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['brain','rsna','vincxr','lag']:
            self.G = Generator(self.g_conv_dim, 0, self.g_repeat_num,device=self.device)
            self.D = Discriminator(self.image_size, self.d_conv_dim, 0, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
                  
        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        best_G_path = os.path.join(self.model_save_dir, 'best_G.ckpt')
        #self.G.load_state_dict(torch.load(best_G_path, map_location=lambda storage, loc: storage))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))

        # if D_path exists, load it
        if os.path.exists(D_path):
            self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))


    def update_lr(self, g_lr,d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr            

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)
                    
    def train(self):

        # Set data loader.
        if self.dataset in ['rsna','vincxr','lag']:
            data_loader = self.data_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixedA, x_fixedB,ano_fixB,label_fix = next(data_iter)
        x_fixedA = x_fixedA.to(self.device)
        x_fixedB = x_fixedB.to(self.device)
        ano_fixB = ano_fixB.to(self.device)
        bs, C, W, H = x_fixedA.size()
        x_fixedA_p = make_window(x_fixedA.permute(0, 2, 3, 1).contiguous(), W//self.num_patch, H//self.num_patch, 0).permute(0, 3, 1, 2)
        x_fixedB_p = make_window(x_fixedB.permute(0, 2, 3, 1).contiguous(), W//self.num_patch, H//self.num_patch, 0).permute(0, 3, 1, 2)
        ano_fixB_p = make_window(ano_fixB.permute(0, 2, 3, 1).contiguous(), W//self.num_patch, H//self.num_patch, 0).permute(0, 3, 1, 2)
        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        best_auc,best_ap = 0.,0.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_realA, x_realB,ano_B,label_B = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_realA, x_realB,ano_B,label_B = next(data_iter)

            x_realA = x_realA.to(self.device)           # Input images.
            x_realB = x_realB.to(self.device) 
            ano_B = ano_B.to(self.device)
            bs, C, W, H = x_realB.size()
            x_realA_p = make_window(x_realA.permute(0, 2, 3, 1).contiguous(), W//self.num_patch, H//self.num_patch, 0).permute(0, 3, 1, 2)
            x_realB_p = make_window(x_realB.permute(0, 2, 3, 1).contiguous(), W//self.num_patch, H//self.num_patch, 0).permute(0, 3, 1, 2)          # Input images.
            ano_B_p = make_window(ano_B.permute(0, 2, 3, 1).contiguous(), W//self.num_patch, H//self.num_patch, 0).permute(0, 3, 1, 2)

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            _, out_src = self.D(x_realB)
            d_loss_real = - torch.mean(out_src)

            # Compute loss with fake images.
            x_fakeA1 = self.G(x_realA_p,bs)
            B_, c, w, h = x_fakeA1.size()
            x_fakeA1 = window_reverse(x_fakeA1.permute(0, 2, 3, 1).contiguous(), w, w * self.num_patch, w * self.num_patch).permute(0, 3, 1, 2)
            _, out_src2 = self.D(x_fakeA1.detach())
            d_loss_fake =torch.mean(out_src2)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_realB.size(0), 1, 1, 1).to(self.device)
            x_hat2 = (alpha * x_realB.data + (1 - alpha) * x_fakeA1.data).requires_grad_(True)
            _, out_src2 = self.D(x_hat2)
            d_loss_gp = self.gradient_penalty(out_src2, x_hat2)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:
                # Original-to-original domain.
                
                x_fakeB3 = self.G(x_realB_p,bs)
                x_fakeB3 = window_reverse(x_fakeB3.permute(0, 2, 3, 1).contiguous(), w, w * self.num_patch, w * self.num_patch).permute(0, 3, 1, 2)
                g_loss_id = torch.mean(torch.abs(x_realB - x_fakeB3))


                # Original-to-target domain.
                x_fakeB1 = self.G(x_realA_p,bs)
                x_fakeB1 = window_reverse(x_fakeB1.permute(0, 2, 3, 1).contiguous(), w, w * self.num_patch, w * self.num_patch).permute(0, 3, 1, 2)

                x_fakeaB = self.G(ano_B_p,bs)
                x_fakeaB = window_reverse(x_fakeaB.permute(0, 2, 3, 1).contiguous(), w, w * self.num_patch, w * self.num_patch).permute(0, 3, 1, 2)
                
                g_loss_mse = torch.mean((x_realB - x_fakeaB) ** 2)

                _, out_src2 = self.D(x_fakeB1)
                g_loss_fake = - torch.mean(out_src2)

                g_loss = g_loss_fake + self.lambda_id * g_loss_id + g_loss_mse

                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_id'] = g_loss_id.item()
                loss['G/loss_mse'] = g_loss_mse.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                auc,ap = self.test_single_model(i+1)
                if auc >= best_auc and  ap >= best_ap:
                    best_auc,best_ap = auc,ap
                    best_model_path = os.path.join(self.model_save_dir,'best_G1.ckpt')
                    torch.save(self.G.state_dict(),best_model_path)
                    print('Save the best generator model  checkpoints into{}...'.format(self.model_save_dir))
                log = "Elapsed [{}], Iteration [{}/{}] AUC:[{:.4f}] AP:[{:.4f}]".format(et, i+1, self.num_iters,auc,ap)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)
                print('best auc:{:4f} ap:{:4f}'.format(best_auc,best_ap))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixedA]
                    fake1 = self.G(x_fixedA_p,bs)
                    fake1 = window_reverse(fake1.permute(0, 2, 3, 1).contiguous(), w, w * self.num_patch, w * self.num_patch).permute(0, 3, 1, 2)
                    x_fake_list.append(fake1)
                    x_fake_list.append(x_fixedB)

                    fake3= self.G(x_fixedB_p,bs)
                    fake3 = window_reverse(fake3.permute(0, 2, 3, 1).contiguous(), w, w * self.num_patch, w * self.num_patch).permute(0, 3, 1, 2)
                    x_fake_list.append(fake3)                    
                    x_concat = torch.cat(x_fake_list, dim=3)

                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

    
    def create_binary(self, pred_label):
        pred_label_res = torch.zeros_like(pred_label)
        for i in range(self.batch_size):
            threshold = torch.kthvalue(pred_label[i].flatten(), int(pred_label[i].numel() * 0.75)).values.item()
            binary_label = (pred_label[i] >= threshold).float()
            pred_label_res[i] = binary_label
        return pred_label_res



    def Find_Optimal_Cutoff(self, target, predicted):
        """ Find the optimal probability cutoff point for a classification model related to event rate
        Parameters
        ----------
        target : Matrix with dependent or target data, where rows are observations

        predicted : Matrix with predicted data, where rows are observations

        Returns
        -------     
        list type, with optimal cutoff value
            
        """
        fpr, tpr, threshold = roc_curve(target, predicted)
        i = np.arange(len(tpr)) 
        roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
        roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

        return list(roc_t['threshold'])
    
    def test_single_model(self,iters):
        self.restore_model(iters)
        from data_loader import get_loader
        data_loader = get_loader(self.dataset,1,self.image_size,self.num_workers,mode='test')

        gt_d = {}
        meanp_d = {}

        with torch.no_grad():
            for i, (x_realA,fname,label) in tqdm(enumerate(data_loader), total=len(data_loader)):
                imgid = fname[0].split('/')[-1].split('__')[0]
                x_realA = x_realA.to(self.device)
                bs,C,W,H = x_realA.shape
                x_realA_p = make_window(x_realA.permute(0, 2, 3, 1).contiguous(), W//self.num_patch, H//self.num_patch, 0).permute(0, 3, 1, 2)

                gt_d[imgid] = label.float()

                # Translate images.
                fake = self.G(x_realA_p)
                B_,c,w,h = fake.shape
                fake = window_reverse(fake.permute(0, 2, 3, 1).contiguous(), w, w * self.num_patch, w * self.num_patch).permute(0, 3, 1, 2)
                diff = torch.abs(x_realA - fake)
                diff /= 2.
                diff = diff.data.cpu().numpy()
                meanp = list(np.mean(diff, axis=(1,2,3)))

                if imgid in meanp_d:
                    meanp_d[imgid] += meanp
                else:
                    meanp_d[imgid] = meanp

        meanp = []
        gt = []
        ks = []

        for k in gt_d.keys():
            ks.append(k)
            gt.append(gt_d[k])
            meanp.append(np.mean(meanp_d[k]))
        gt = np.array([t.item() for t in gt])
        meanp = np.array(meanp)

        thmean = self.Find_Optimal_Cutoff(gt, meanp)[0]
        meanpth = (np.array(meanp)>=thmean)

        meanauc = roc_auc_score(gt, meanp)
        ap = average_precision_score(gt,meanp)

        return meanauc,ap
    
    
    def test(self):
        """Translate images using Brainomaly trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)

        from data_loader import get_loader
        data_loader = get_loader(self.dataset,1,self.image_size,self.num_workers,mode='test')

        gt_d = {}
        meanp_d = {}
        with torch.no_grad():
            for i, (x_realA,fname,label) in tqdm(enumerate(data_loader), total=len(data_loader)):
                imgid = fname[0].split('/')[-1].split('__')[0]
                #print(fname)
                x_realA = x_realA.to(self.device)
                save_list = [x_realA]
                bs,C,W,H = x_realA.shape
                x_realA_p = make_window(x_realA.permute(0, 2, 3, 1).contiguous(), W//self.num_patch, H//self.num_patch, 0).permute(0, 3, 1, 2)

                gt_d[imgid] = label.float()

                # Translate images.

                fake = self.G(x_realA_p)

                B_,c,w,h = fake.shape
                fake = window_reverse(fake.permute(0, 2, 3, 1).contiguous(), w, w * self.num_patch, w * self.num_patch).permute(0, 3, 1, 2)
                '''
                anomaly_map,_ = cal_anomaly_map([x_realA],[fake],x_realA.shape[-1])
                ano_map = min_max_norm(anomaly_map)
                ano_map = cvt2heatmap(ano_map*255)
                img = cv2.cvtColor(x_realA.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = np.uint8(min_max_norm(img)*255)
                img = np.broadcast_to(img[:, :, np.newaxis], (img.shape[0], img.shape[1], 3))
                ano_map = show_cam_on_image(img, ano_map)
                cam_sample_path = os.path.join(self.save_dir,imgid +'_cam.png')
                cv2.imwrite(cam_sample_path, ano_map)

                save_list.append(fake)
                sample_path = os.path.join(self.save_dir,imgid +'.png')
            
                concat_list = torch.cat(save_list, dim=3)
                save_image(self.denorm(concat_list.data.cpu()), sample_path, nrow=1, padding=0)
                '''
                
                diff = torch.abs(x_realA - fake)
                diff /= 2.
                diff = diff.data.cpu().numpy()
                meanp = list(np.mean(diff, axis=(1,2,3)))

                if imgid in meanp_d:
                    meanp_d[imgid] += meanp
                else:
                    meanp_d[imgid] = meanp

        meanp = []
        gt = []
        ks = []

        for k in gt_d.keys():
            ks.append(k)
            gt.append(gt_d[k])
            meanp.append(np.mean(meanp_d[k]))
        gt = np.array([t.item() for t in gt])
        meanp = np.array(meanp)

        thmean = self.Find_Optimal_Cutoff(gt, meanp)[0]
        meanpth = (np.array(meanp)>=thmean)

        meanauc = roc_auc_score(gt, meanp)
        ap = average_precision_score(gt,meanp)

        print(f"Model Iter {self.test_iters} AUC: {round(meanauc, 4)} AP:{round(ap,4)}") 

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def make_window(x, window_size, stride=1, padding=0):
    """
    Args:
        x: (B, W, H, C)
        window_size (int): window size

    Returns:
        windows: (B*N,  ws**2, C)
    """
    x = x.permute(0, 3, 1, 2).contiguous()
    B, C, W, H = x.shape
    windows = F.unfold(x, window_size, padding=padding, stride=stride) # B, C*N, #of windows
    windows = windows.view(B, C, window_size**2, -1) #   B, C, ws**2, N
    windows = windows.permute(0, 3, 2, 1).contiguous().view(-1, window_size, window_size, C) # B*N, ws**2, C
    
    return windows

def show_cam_on_image(img, anomaly_map):
    #if anomaly_map.shape != img.shape:
    #    anomaly_map = cv2.applyColorMap(np.uint8(anomaly_map), cv2.COLORMAP_JET)
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def cal_anomaly_map(fs_list, ft_list, out_size=64):
 
    anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        #fs_norm = F.normalize(fs, p=2)
        #ft_norm = F.normalize(ft, p=2)
        a_map = torch.abs(fs-ft)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)

        anomaly_map += a_map
    return anomaly_map, a_map_list

