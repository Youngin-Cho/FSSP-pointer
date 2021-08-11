import os
import torch
import torch.nn as nn
import torch.optim as optim

from time import time
from datetime import datetime

from agent.actor import PtrNet1
from agent.critic import PtrNet2
from environment.env import PanelBlockShop


# torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True


def initialize_model(env, params, log_path=None):
    date = datetime.now().strftime('%m%d_%H_%M')
    param_path = params["log_dir"] + '/%s_%s_param.csv' % (date, "initialization")
    print(f'generate {param_path}')
    with open(param_path, 'w') as f:
        f.write(''.join('%s,%s\n' % item for item in params.items()))

    act_model = PtrNet1(params)

    if params["optimizer"] == 'Adam':
        act_optim = optim.Adam(act_model.parameters(), lr=params["lr"])
    elif params["optimizer"] == "RMSProp":
        act_optim = optim.RMSprop(act_model.parameters(), lr=params["lr"])
    if params["is_lr_decay"]:
        act_lr_scheduler = optim.lr_scheduler.StepLR(act_optim, step_size=params["lr_decay_step"],
                                                     gamma=params["lr_decay"])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    act_model = act_model.to(device)
    ce_loss = nn.NLLLoss()
    ave_act_loss = 0.0
    ave_makespan = 0.0

    t1 = time()
    for s in range(params["step"]):
        inputs, labels = env.generate_data(params["batch_size"], use_label=True)

        pred_seq, _, log_p = act_model(inputs, device, labels)
        real_makespan = env.stack_makespan(inputs, pred_seq)

        log_p_unrolled = log_p.view(-1, log_p.size(-1))
        labels_unrolled = labels.view(-1).long()
        act_loss = ce_loss(log_p_unrolled, labels_unrolled)
        act_optim.zero_grad()
        act_loss.backward()
        act_optim.step()
        nn.utils.clip_grad_norm_(act_model.parameters(), max_norm=1., norm_type=2)
        if params["is_lr_decay"]:
            act_lr_scheduler.step()
        ave_act_loss += act_loss.item()

        ave_makespan += real_makespan.mean().item()

        if s % params["log_step"] == 0:
            t2 = time()
            print('step:%d/%d, actic loss:%1.3f, L:%1.3f, %dmin%dsec' % (
                s, params["step"], ave_act_loss / (s + 1), ave_makespan / (s + 1), (t2 - t1) // 60, (t2 - t1) % 60))
            if log_path is None:
                log_path = params["log_dir"] + '/%s_initialization.csv' % date
                with open(log_path, 'w') as f:
                    f.write('step,actic loss,average distance,time\n')
            else:
                with open(log_path, 'a') as f:
                    f.write('%d,%1.4f,%1.4f,%dmin%dsec\n' % (s, ave_act_loss / (s + 1), ave_makespan / (s + 1),
                                                             (t2 - t1) // 60, (t2 - t1) % 60))
            t1 = time()

        if s % params["save_step"] == 0:
            torch.save(act_model.state_dict(), params["model_dir"] +  '/%s_step%d_act.pt' % (date, s))
            print('save model...')


if __name__ == '__main__':

    load_model = False

    log_dir = "./result/log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    model_dir = "./result/model/sl"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    params = {
        "num_of_process": 6,
        "num_of_blocks": 40,
        "step": 10000,
        "log_step": 10,
        "log_dir": log_dir,
        "save_step": 1000,
        "model_dir": model_dir,
        "batch_size": 32,
        "n_embedding": 512,
        "n_hidden": 512,
        "init_min": -0.08,
        "init_max": 0.08,
        "clip_logits": 10,
        "softmax_T": 1.0,
        "decode_type": "sampling",
        "optimizer": "Adam",
        "n_glimpse": 1,
        "n_process": 3,
        "lr": 1e-3,
        "is_lr_decay": True,
        "lr_decay": 0.98,
        "lr_decay_step": 2000,
        "use_critic": False,
        "load_model": False
    }

    env = PanelBlockShop(params["num_of_process"], params["num_of_blocks"])
    initialize_model(env, params)