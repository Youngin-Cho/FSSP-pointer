import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import scipy.stats as stats

from torch.utils.data import DataLoader
from time import time
from datetime import datetime

from actor import PtrNet1
from critic import PtrNet2
from env import PanelBlockShop
from config import Config, load_pkl, pkl_parser
from data import Generator

# torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True

shape = [0.543, 0.525, 0.196, 0.451, 0.581, 0.432]
scale = [2.18, 2.18, 0.518, 2.06, 1.79, 2.10]
mean = [np.round(stats.lognorm.mean(shape[i], scale=scale[i]), 3) for i in range(6)]
std = [np.round(stats.lognorm.std(shape[i], scale=scale[i]), 3) for i in range(6)]


def train_model(cfg, env, log_path=None):
    date = datetime.now().strftime('%m%d_%H_%M')
    if cfg.islogger:
        param_path = cfg.log_dir + '%s_%s_param.csv' % (date, cfg.task)  # cfg.log_dir = ./Csv/
        print(f'generate {param_path}')
        with open(param_path, 'w') as f:
            f.write(''.join('%s,%s\n' % item for item in vars(cfg).items()))

    act_model = PtrNet1(cfg)
    if cfg.optim == 'Adam':
        act_optim = optim.Adam(act_model.parameters(), lr=cfg.lr)
    if cfg.is_lr_decay:
        act_lr_scheduler = optim.lr_scheduler.StepLR(act_optim, step_size=cfg.lr_decay_step, gamma=cfg.lr_decay)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    act_model = act_model.to(device)

    if cfg.mode == 'train':
        cri_model = PtrNet2(cfg)
        if cfg.optim == 'Adam':
            cri_optim = optim.Adam(cri_model.parameters(), lr=cfg.lr)
        if cfg.is_lr_decay:
            cri_lr_scheduler = optim.lr_scheduler.StepLR(cri_optim, step_size=cfg.lr_decay_step, gamma=cfg.lr_decay)
        cri_model = cri_model.to(device)
        ave_cri_loss = 0.

    mse_loss = nn.MSELoss()
    dataset = Generator(cfg, env)
    dataloader = DataLoader(dataset, batch_size=cfg.batch, shuffle=True)

    ave_act_loss, ave_C = 0., 0.
    min_C, cnt = 1e7, 0
    t1 = time()
    for i, inputs in enumerate(dataloader):
        inputs = inputs.to(device)
        inputs_network = inputs
        for j in range(6):
            inputs_network[:,:,j] = (inputs[:,j] - mean[j]) / std[j]

        pred_seq, ll = act_model(inputs_network, device)
        real_C = env.stack_C(inputs, pred_seq)
        if cfg.mode == 'train':
            pred_C = cri_model(inputs_network, device)
            cri_loss = mse_loss(pred_C, real_C.detach())
            cri_optim.zero_grad()
            cri_loss.backward()
            nn.utils.clip_grad_norm_(cri_model.parameters(), max_norm=1., norm_type=2)
            cri_optim.step()
            if cfg.is_lr_decay:
                cri_lr_scheduler.step()
        elif cfg.mode == 'train_emv':
            if i == 0:
                C = real_C.detach().mean()
            else:
                C = (C * 0.9) + (0.1 * real_C.detach().mean())
            pred_C = C

        adv = real_C.detach() - pred_C.detach()
        act_loss = (adv * ll).mean()
        act_optim.zero_grad()
        act_loss.backward()
        nn.utils.clip_grad_norm_(act_model.parameters(), max_norm=1., norm_type=2)
        act_optim.step()
        if cfg.is_lr_decay:
            act_lr_scheduler.step()

        ave_act_loss += act_loss.item()
        if cfg.mode == 'train':
            ave_cri_loss += cri_loss.item()
        ave_C += real_C.mean().item()

        if i % cfg.log_step == 0:
            t2 = time()
            if cfg.mode == 'train':
                print('step:%d/%d, actic loss:%1.3f, critic loss:%1.3f, L:%1.3f, %dmin%dsec' % (
                i, cfg.steps, ave_act_loss / (i + 1), ave_cri_loss / (i + 1), ave_C / (i + 1), (t2 - t1) // 60,
                (t2 - t1) % 60))
                if cfg.islogger:
                    if log_path is None:
                        log_path = cfg.log_dir + '%s_%s_train.csv' % (date, cfg.task)  # cfg.log_dir = ./Csv/
                        with open(log_path, 'w') as f:
                            f.write('step,actic loss,critic loss,average distance,time\n')
                    else:
                        with open(log_path, 'a') as f:
                            f.write('%d,%1.4f,%1.4f,%1.4f,%dmin%dsec\n' % (
                            i, ave_act_loss / (i + 1), ave_cri_loss / (i + 1), ave_C / (i + 1), (t2 - t1) // 60,
                            (t2 - t1) % 60))

            elif cfg.mode == 'train_emv':
                print('step:%d/%d, actic loss:%1.3f, C:%1.3f, %dmin%dsec' % (
                i, cfg.steps, ave_act_loss / (i + 1), ave_C / (i + 1), (t2 - t1) // 60, (t2 - t1) % 60))
                if cfg.islogger:
                    if log_path is None:
                        log_path = cfg.log_dir + '%s_%s_train_emv.csv' % (date, cfg.task)  # cfg.log_dir = ./Csv/
                        with open(log_path, 'w') as f:
                            f.write('step,actic loss,average distance,time\n')
                    else:
                        with open(log_path, 'a') as f:
                            f.write('%d,%1.4f,%1.4f,%dmin%dsec\n' % (
                            i, ave_act_loss / (i + 1), ave_C / (i + 1), (t2 - t1) // 60, (t2 - t1) % 60))
            if (ave_C / (i + 1) < min_C):
                # cnt = 0
                min_C = ave_C / (i + 1)
            # else:
            #     cnt += 1
            #     print(f'cnt: {cnt}/100')
            #     if (cnt >= 100):
            #         print('early stop, average cost cant decrease anymore')
            #         if log_path is not None:
            #             with open(log_path, 'a') as f:
            #                 f.write('\nearly stop')
            #         break
            t1 = time()

        if cfg.issaver and i % 1000 == 0:
            torch.save(act_model.state_dict(),
                       cfg.model_dir + '%s_%s_step%d_act.pt' % (cfg.task, date, i))  # 'cfg.model_dir = ./Pt/'
            print('save model...')


if __name__ == '__main__':
    cfg = load_pkl(pkl_parser().path)
    env = PanelBlockShop(cfg)

    if cfg.mode in ['train', 'train_emv']:
        # train_emv --> exponential moving average, not use critic model
        train_model(cfg, env)
    else:
        raise NotImplementedError('train and train_emv only, specify train pkl file')