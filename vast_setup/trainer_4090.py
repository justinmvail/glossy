import torch
from tensorboardX import SummaryWriter
from parse_config import cfg
import os
import subprocess
import torchvision
from tqdm import tqdm
from data_loader.loader import ContentData
import torch.distributed as dist

# Google Drive folder ID for checkpoint uploads
GDRIVE_FOLDER_ID = "1UY61ytrE6ec-OBdMESZvcpD9gcVsz_ad"

class Trainer:
    def __init__(self, diffusion, unet, vae, criterion, optimizer, data_loader, logs, valid_data_loader=None, device=None, ocr_model=None, ctc_loss=None):
        self.model = unet
        self.diffusion = diffusion
        self.vae = vae
        self.recon_criterion = criterion["recon"]
        self.nce_criterion = criterion["nce"]
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.tb_summary = SummaryWriter(logs["tboard"])
        self.save_model_dir = logs["model"]
        self.save_sample_dir = logs["sample"]
        self.device = device

    def _train_iter(self, data, step, pbar):
        self.model.train()
        images = data["img"].to(self.device)
        style_ref = data["style"].to(self.device)
        laplace_ref = data["laplace"].to(self.device)
        content_ref = data["content"].to(self.device)
        wid = data["wid"].to(self.device)
        with torch.no_grad():
            images = self.vae.encode(images).latent_dist.sample() * 0.18215
        t = self.diffusion.sample_timesteps(images.shape[0]).to(self.device)
        x_t, noise = self.diffusion.noise_images(images, t)
        predicted_noise, high_nce_emb, low_nce_emb = self.model(x_t, t, style_ref, laplace_ref, content_ref, tag="train")
        recon_loss = self.recon_criterion(predicted_noise, noise)
        high_nce_loss = self.nce_criterion(high_nce_emb, labels=wid)
        low_nce_loss = self.nce_criterion(low_nce_emb, labels=wid)
        loss = recon_loss + high_nce_loss + low_nce_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        if dist.get_rank() == 0:
            self.tb_summary.add_scalars("loss", {"reconstruct_loss": recon_loss.item(), "high_nce_loss": high_nce_loss.item(), "low_nce_loss": low_nce_loss.item()}, step)
            pbar.set_postfix(mse="%.6f" % recon_loss.item())

    def _save_images(self, images, path):
        grid = torchvision.utils.make_grid(images)
        torchvision.transforms.ToPILImage()(grid).save(path)

    @torch.no_grad()
    def _valid_iter(self, epoch):
        self.model.eval()
        test_data = next(iter(self.valid_data_loader))
        style_ref = test_data["style"].to(self.device)
        laplace_ref = test_data["laplace"].to(self.device)
        load_content = ContentData()
        for text in ["getting", "both", "success"]:
            text_ref = load_content.get_content(text).to(self.device).repeat(style_ref.shape[0], 1, 1, 1)
            x = torch.randn((text_ref.shape[0], 4, style_ref.shape[2]//8, (text_ref.shape[1]*32)//8)).to(self.device)
            preds = self.diffusion.ddim_sample(self.model, self.vae, style_ref.shape[0], x, style_ref, laplace_ref, text_ref)
            self._save_images(preds, os.path.join(self.save_sample_dir, f"epoch-{epoch}-{text}-process-{dist.get_rank()}.png"))

    def _upload_to_gdrive(self, filepath, delete_old=True):
        """Upload checkpoint to Google Drive and optionally delete old ones"""
        try:
            filename = os.path.basename(filepath)
            print(f"Uploading {filename} to Google Drive...")

            # Upload new checkpoint
            result = subprocess.run(
                ["gdrive", "files", "upload", "--parent", GDRIVE_FOLDER_ID, filepath],
                capture_output=True, text=True, timeout=600
            )

            if result.returncode == 0:
                print(f"Uploaded {filename} to Google Drive")

                # Delete old checkpoints from Drive (keep only latest)
                if delete_old:
                    # List files in folder
                    list_result = subprocess.run(
                        ["gdrive", "files", "list", "--parent", GDRIVE_FOLDER_ID],
                        capture_output=True, text=True
                    )
                    # Find and delete old ckpt files (not the current one)
                    for line in list_result.stdout.split('\n'):
                        if '-ckpt.pt' in line and filename not in line:
                            file_id = line.split()[0]
                            subprocess.run(["gdrive", "files", "delete", file_id], capture_output=True)
                            print(f"Deleted old checkpoint from Drive")
            else:
                print(f"Upload failed: {result.stderr}")
        except Exception as e:
            print(f"Upload error (non-fatal): {e}")

    def train(self):
        for epoch in range(cfg.SOLVER.EPOCHS):
            self.data_loader.sampler.set_epoch(epoch)
            print(f"Epoch:{epoch} of process {dist.get_rank()}")
            dist.barrier()
            pbar = tqdm(self.data_loader, leave=False) if dist.get_rank() == 0 else self.data_loader
            for step, data in enumerate(pbar):
                self._train_iter(data, epoch * len(self.data_loader) + step, pbar)
            if (epoch+1) > cfg.TRAIN.SNAPSHOT_BEGIN and (epoch+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0 and dist.get_rank() == 0:
                ckpt_path = os.path.join(self.save_model_dir, str(epoch)+"-ckpt.pt")
                torch.save(self.model.module.state_dict(), ckpt_path)
                self._upload_to_gdrive(ckpt_path, delete_old=True)
            if self.valid_data_loader and (epoch+1) > cfg.TRAIN.VALIDATE_BEGIN and (epoch+1) % cfg.TRAIN.VALIDATE_ITERS == 0:
                self._valid_iter(epoch)
