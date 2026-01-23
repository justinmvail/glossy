import argparse
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
from utils.util import fix_seed, load_specific_dict
from utils.logger import set_log
from data_loader.loader import IAMDataset
import torch
from trainer.trainer import Trainer
from models.unet import UNetModel
from torch import optim
import torch.nn as nn
from models.diffusion import Diffusion, EMA
import copy
from diffusers import AutoencoderKL
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from models.loss import SupConLoss

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main(opt):
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()
    fix_seed(cfg.TRAIN.SEED)
    logs = set_log(cfg.OUTPUT_DIR, opt.cfg_file, opt.log_name)
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(opt.device, local_rank)

    train_dataset = IAMDataset(cfg.DATA_LOADER.IAMGE_PATH, cfg.DATA_LOADER.STYLE_PATH, cfg.DATA_LOADER.LAPLACE_PATH, cfg.TRAIN.TYPE)
    print("number of training images:", len(train_dataset))
    train_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.TRAIN.IMS_PER_BATCH, drop_last=False, collate_fn=train_dataset.collate_fn_, num_workers=cfg.DATA_LOADER.NUM_THREADS, pin_memory=True, persistent_workers=True, prefetch_factor=4, sampler=train_sampler)

    test_dataset = IAMDataset(cfg.DATA_LOADER.IAMGE_PATH, cfg.DATA_LOADER.STYLE_PATH, cfg.DATA_LOADER.LAPLACE_PATH, cfg.TEST.TYPE)
    test_sampler = DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.TEST.IMS_PER_BATCH, drop_last=False, collate_fn=test_dataset.collate_fn_, pin_memory=True, num_workers=cfg.DATA_LOADER.NUM_THREADS, persistent_workers=True, prefetch_factor=4, sampler=test_sampler)

    unet = UNetModel(in_channels=cfg.MODEL.IN_CHANNELS, model_channels=cfg.MODEL.EMB_DIM, out_channels=cfg.MODEL.OUT_CHANNELS, num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS, attention_resolutions=(1,1), channel_mult=(1,1), num_heads=cfg.MODEL.NUM_HEADS, context_dim=cfg.MODEL.EMB_DIM, use_checkpoint=False).to(device)
    print("Gradient checkpointing DISABLED")

    if len(opt.one_dm) > 0:
        unet.load_state_dict(torch.load(opt.one_dm, map_location=torch.device("cpu")))
    if len(opt.feat_model) > 0:
        checkpoint = torch.load(opt.feat_model, map_location=torch.device("cpu"))
        checkpoint["conv1.weight"] = checkpoint["conv1.weight"].mean(1).unsqueeze(1)
        unet.mix_net.Feat_Encoder.load_state_dict(checkpoint, strict=False)

    unet = DDP(unet, device_ids=[local_rank])
    criterion = dict(nce=SupConLoss(contrast_mode="all"), recon=nn.MSELoss())
    optimizer = optim.AdamW(unet.parameters(), lr=cfg.SOLVER.BASE_LR)
    print("Using AdamW optimizer")

    diffusion = Diffusion(device=device, noise_offset=opt.noise_offset)
    vae = AutoencoderKL.from_pretrained(opt.stable_dif_path, subfolder="vae")
    vae.requires_grad_(False)
    vae = vae.to(device)
    print("AMP + TF32 enabled")

    trainer = Trainer(diffusion, unet, vae, criterion, optimizer, train_loader, logs, test_loader, device)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stable_dif_path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--cfg", dest="cfg_file", default="configs/IAM64_scratch.yml")
    parser.add_argument("--feat_model", dest="feat_model", default="")
    parser.add_argument("--one_dm", dest="one_dm", default="")
    parser.add_argument("--log", default="debug", dest="log_name", required=False)
    parser.add_argument("--noise_offset", default=0, type=float)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--local_rank", type=int, default=0)
    opt = parser.parse_args()
    main(opt)
