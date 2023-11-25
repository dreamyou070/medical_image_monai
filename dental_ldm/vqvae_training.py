from monai import transforms
import argparse, wandb
from random import seed
from helpers import *
from torchvision import transforms
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data_module import SYDataLoader, SYDataset
from monai.utils import first
from setproctitle import *
from generative.networks.nets import VQVAE
from tqdm import tqdm
from torch.nn import L1Loss
import torchvision.transforms as torch_transforms
from PIL import Image

def main(args):

    print(f'\n step 1. setting')
    if args.process_title:
        setproctitle(args.process_title)
    else:
        setproctitle('parksooyeon')

    print(f' (1.1) wandb')
    os.environ["WANDB_DATA_DIR"] = args.experiment_dir
    wandb.login(key=args.wandb_api_key)
    wandb.init(project=args.wandb_project_name, name=args.wandb_run_name,
               dir=args.experiment_dir)

    print(f' (1.2) seed and device')
    seed(args.seed)
    device = args.device

    print(f' (1.3) saving configuration')
    experiment_dir = args.experiment_dir
    os.makedirs(experiment_dir, exist_ok=True)
    var_args = vars(args)
    with open(os.path.join(experiment_dir, "config.txt"), "w") as f:
        for key in sorted(var_args.keys()):
            f.write(f"{key}: {var_args[key]}\n")

    print(f'\n step 2. dataset and dataloatder')
    w, h = int(args.img_size.split(',')[0].strip()), int(args.img_size.split(',')[1].strip())
    train_transforms = transforms.Compose([transforms.Resize((w, h), transforms.InterpolationMode.BILINEAR),
                                           transforms.ToTensor()])
    train_ds = SYDataset(data_folder=args.train_data_folder,
                         transform=train_transforms,
                         base_mask_dir=args.train_mask_dir,
                         image_size=(w, h))
    training_dataset_loader = SYDataLoader(train_ds,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=4,
                                           persistent_workers=True)
    check_data = first(training_dataset_loader)
    # ## Prepare validation set data loader
    val_transforms = transforms.Compose([transforms.Resize((w, h), transforms.InterpolationMode.BILINEAR),
                                         transforms.ToTensor()])
    val_ds = SYDataset(data_folder=args.val_data_folder,
                       transform=val_transforms,
                       base_mask_dir=args.val_mask_dir, image_size=(w, h))
    test_dataset_loader = SYDataLoader(val_ds,
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       num_workers=4,
                                       persistent_workers=True)

    print(f'\n step 3. model')
    model = VQVAE(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_channels=(256, 256),
        num_res_channels=256,
        num_res_layers=2,
        downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1)),
        upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
        num_embeddings=256,
        embedding_dim=32,)
    model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
    l1_loss = L1Loss()

    print(f'\n step 4. model training')
    n_epochs = 100

    val_recon_epoch_loss_list = []
    intermediary_images = []
    n_example_images = 4

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(training_dataset_loader), total=len(training_dataset_loader), ncols=110)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            images = batch["image_info"].to(device)
            normal_info = batch['normal']  # if 1 = normal, 0 = abnormal
            normal_images = images[normal_info == 1]
            if normal_images.shape[0] > 0:
                images = normal_images
                optimizer.zero_grad(set_to_none=True)
                # ---------------------------------------------------------------------------------------------------------------
                # model outputs reconstruction and the quantization error
                reconstruction, quantization_loss = model(images=images)
                recons_loss = l1_loss(reconstruction.float(), images.float())
                loss = recons_loss + quantization_loss
                wandb.log({"recons_loss": recons_loss.item(),
                           "quantization_loss": quantization_loss.item()})
                loss.backward()
                optimizer.step()
                epoch_loss += recons_loss.item()
                progress_bar.set_postfix({"recons_loss": epoch_loss / (step + 1), "quantization_loss": quantization_loss.item() / (step + 1)})

        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_step, batch in enumerate(test_dataset_loader, start=1):
                    images = batch["image_info"].to(device)
                    normal_info = batch['normal']
                    normal_images = images[normal_info == 1]
                    if normal_images.shape[0] > 0:
                        images = normal_images
                        reconstruction, quantization_loss = model(images=images)
                        if val_step == 1:
                            # [Batch, W, H]
                            trg_img = reconstruction[:n_example_images, 0]
                            num = trg_img.shape[0]
                            for i in range(num):
                                intermediary_images.append(wandb.Image(trg_img[i].cpu().numpy()))
                                org_img = images[i].squeeze()
                                org_img = torch_transforms.ToPILImage()(org_img.unsqueeze(0))
                                recon = reconstruction[0].squeeze()
                                recon = torch_transforms.ToPILImage()(recon.unsqueeze(0))
                                new = Image.new('RGB', (org_img.width + recon.width, org_img.height))
                                new.paste(org_img, (0, 0))
                                new.paste(recon, (org_img.width, 0))
                                loading_image = wandb.Image(new, caption=f"(real-recon) epoch {epoch + 1} ")
                                wandb.log({"vqvae inference": loading_image})
                        recons_loss = l1_loss(reconstruction.float(), images.float())
                        val_loss += recons_loss.item()
            val_loss /= val_step
            wandb.log({"val_recons_loss": val_loss})
        # save model
        if epoch > args.model_save_base_epoch:
            model_save_dir = os.path.join(experiment_dir, f'vqvae')
            os.makedirs(model_save_dir, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(model_save_dir, f'vqvae_{epoch}.pth'))

    progress_bar.close()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # step 1. wandb login
    parser.add_argument("--process_title", type=str, default='parksooyeon')
    parser.add_argument("--wandb_api_key", type=str, default='3a3bc2f629692fa154b9274a5bbe5881d47245dc')
    parser.add_argument("--wandb_project_name", type=str, default='dental_experiment')
    parser.add_argument("--wandb_run_name", type=str, default='hand_1000_64res')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str)
    parser.add_argument('--experiment_dir', type=str,
                        default=f'/data7/sooyeon/medical_image/anoddpm_result/20231119_dental_test')

    # step 2. dataset and dataloatder
    parser.add_argument('--train_data_folder', type=str)
    parser.add_argument('--train_mask_dir', type=str)
    parser.add_argument('--val_data_folder', type=str)
    parser.add_argument('--val_mask_dir', type=str)
    parser.add_argument('--img_size', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--model_save_base_epoch', type=int, default=50)

    parser.add_argument('--val_interval', type=int, default = 10)
    args = parser.parse_args()
    main(args)
