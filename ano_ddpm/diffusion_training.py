import argparse
import collections
import copy
import sys
import matplotlib.pyplot as plt
from matplotlib import animation
import time
from random import seed
from torch import optim
from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
from helpers import *
from tqdm import tqdm
from monai import transforms
import numpy  as np
from monai.data import DataLoader, Dataset
from monai.utils import first
from UNet import UNetModel, update_ema_params

torch.cuda.empty_cache()
def training_outputs(diffusion, x, est, noisy, epoch, row_size, ema, args,
                     save_imgs=False, save_vids=False):
    try:
        os.makedirs(f'./diffusion-videos/ARGS={args["arg_num"]}')
        os.makedirs(f'./diffusion-training-images/ARGS={args["arg_num"]}')
    except OSError:
        pass
    if save_imgs:
        if epoch % 100 == 0 or epoch < 2:
            # for a given t, output x_0, & prediction of x_(t-1), and x_0
            noise = torch.rand_like(x)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=x.device)
            x_t = diffusion.sample_q(x, t, noise)
            temp = diffusion.sample_p(ema, x_t, t)
            out = torch.cat((x[:row_size, ...].cpu(), temp["sample"][:row_size, ...].cpu(),
                     temp["pred_x_0"][:row_size, ...].cpu()))
            plt.title(f'real,sample,prediction x_0-{epoch}epoch')
        else:
            # for a given t, output x_0, x_t, & prediction of noise in x_t & MSE
            out = torch.cat(
                    (x[:row_size, ...].cpu(), noisy[:row_size, ...].cpu(), est[:row_size, ...].cpu(),
                     (est - noisy).square().cpu()[:row_size, ...])
                    )
            plt.title(f'real,noisy,noise prediction,mse-{epoch}epoch')
        plt.rcParams['figure.dpi'] = 150
        plt.grid(False)
        plt.imshow(gridify_output(out, row_size), cmap='gray')
        img_save_dir = f'./diffusion-training-images/ARGS={args["arg_num"]}/EPOCH={epoch}.png'
        print(f'saving image to {img_save_dir}')
        plt.savefig(img_save_dir)
        plt.clf()
    if save_vids:
        fig, ax = plt.subplots()
        if epoch % 500 == 0 or epoch < 2 :
            plt.rcParams['figure.dpi'] = 200
            if epoch % 1000 == 0:
                out = diffusion.forward_backward(ema, x, "half", args['sample_distance'] // 2, denoise_fn="noise_fn")
            else:
                out = diffusion.forward_backward(ema, x, "half", args['sample_distance'] // 4, denoise_fn="noise_fn")
            imgs = [[ax.imshow(gridify_output(x, row_size), animated=True)] for x in out]
            ani = animation.ArtistAnimation(fig, imgs, interval=50, blit=True,repeat_delay=1000)

            ani.save(f'./diffusion-videos/ARGS={args["arg_num"]}/sample-EPOCH={epoch}.mp4')

    plt.close('all')

def main(args) :

    print(f'\n step 1. setting')
    print(f' (1.1) base setting')
    device = args['device']
    seed(args['seed'])
    print(f' (1.2) saving configuration')
    experiment_dir = args['experiment_dir']
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, "config.txt"), "w") as f:
        for key in sorted(args.keys()):
            f.write(f"{key}: {args[key]}\n")

    print(f'\n step 2. check file and the argument')
    print(f' - args : {args}')
    in_channels = args["in_channels"]

    """
    print(f'\n step 4. dataset')
    if args["dataset"].lower() == "cifar":
        training_dataset_loader_, testing_dataset_loader_ = dataset.load_CIFAR10(args, True), dataset.load_CIFAR10(args, False)
        training_dataset_loader = dataset.cycle(training_dataset_loader_)
        testing_dataset_loader = dataset.cycle(testing_dataset_loader_)
    elif args["dataset"].lower() == "carpet":
        training_dataset = dataset.DAGM("DATASETS/CARPET/Class1", False, args["img_size"], False)
        training_dataset_loader = dataset.init_dataset_loader(training_dataset, args)
        testing_dataset = dataset.DAGM("DATASETS/CARPET/Class1", True, args["img_size"], False)
        testing_dataset_loader = dataset.init_dataset_loader(testing_dataset, args)
    elif args["dataset"].lower() == "leather":
        if in_channels == 3:
            training_dataset = dataset.MVTec("DATASETS/leather", anomalous=False, img_size=args["img_size"], rgb=True)
            testing_dataset = dataset.MVTec("DATASETS/leather", anomalous=True, img_size=args["img_size"], rgb=True, include_good=True)
        else:
            training_dataset = dataset.MVTec("DATASETS/leather", anomalous=False, img_size=args["img_size"], rgb=False)
            testing_dataset = dataset.MVTec("DATASETS/leather", anomalous=True, img_size=args["img_size"], rgb=False, include_good=True)
        training_dataset_loader = dataset.init_dataset_loader(training_dataset, args)
        testing_dataset_loader = dataset.init_dataset_loader(testing_dataset, args)
    else:
        # load NFBS dataset
        training_dataset, testing_dataset = dataset.init_datasets(ROOT_DIR, args)
        training_dataset_loader = dataset.init_dataset_loader(training_dataset, args)
        testing_dataset_loader = dataset.init_dataset_loader(testing_dataset, args)
    """
    print(f'\n step 3. dataset and dataloatder')
    train_datas = os.listdir(args['train_data_folder'])
    val_datas = os.listdir(args['val_data_folder'])
    train_datalist = [{"image": os.path.join(args['train_data_folder'], train_data)} for train_data in train_datas]
    w,h = args['img_size'][0],args['img_size'][1]
    train_transforms = transforms.Compose([transforms.LoadImaged(keys=["image"]),
                                           transforms.EnsureChannelFirstd(keys=["image"]),
                                           transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
                                           transforms.RandAffined(keys=["image"],
                                                                  rotate_range=[(-np.pi / 36, np.pi / 36),(-np.pi / 36, np.pi / 36)],
                                                                  translate_range=[(-1, 1), (-1, 1)],
                                                                  scale_range=[(-0.05, 0.05), (-0.05, 0.05)],
                                                                  spatial_size=[w,h],
                                                                  padding_mode="zeros",
                                                                  prob=0.5, ), ])
    train_ds = Dataset(data=train_datalist, transform=train_transforms)
    training_dataset_loader = DataLoader(train_ds,
                                         batch_size=args['Batch_Size'],
                                         shuffle=True, num_workers=4, persistent_workers=True)
    check_data = first(training_dataset_loader)
    # ## Prepare validation set data loader
    val_datalist = [{"image": os.path.join(args['val_data_folder'], val_data)} for val_data in val_datas]
    val_transforms = transforms.Compose([transforms.LoadImaged(keys=["image"]),
                                         transforms.EnsureChannelFirstd(keys=["image"]),
                                         transforms.ScaleIntensityRanged(keys=["image"],
                                                                         a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0,
                                                                         clip=True), ])
    val_ds = Dataset(data=val_datalist, transform=val_transforms)
    test_dataset_loader = DataLoader(val_ds,batch_size=args['Batch_Size'],
                                     shuffle=True, num_workers=4, persistent_workers=True)

    print(f'\n step 5. resume or not')
    loaded_model = {}
    resume = args['resume']
    if resume:
        if resume == 1:
            checkpoints = os.listdir(f'model/diff-params-ARGS={args["arg_num"]}/checkpoint')
            checkpoints.sort(reverse=True)
            for i in checkpoints:
                try:
                    file_dir = f"model/diff-params-ARGS={args['arg_num']}/checkpoint/{i}"
                    loaded_model = torch.load(file_dir, map_location=device)
                    break
                except RuntimeError:
                    continue
        else:
            file_dir = f'model/diff-params-ARGS={args["arg_num"]}/params-final.pt'
            loaded_model = torch.load(file_dir, map_location=device)

    print(f'\n step 6. model')
    if args["dataset"].lower() == "cifar" or args["dataset"].lower() == "leather":
        in_channels = 3
    if args["channels"] != "":
        in_channels = args["channels"]

    model = UNetModel(args['img_size'][0],
                      args['base_channels'],
                      channel_mults=args['channel_mults'],
                      dropout=args["dropout"],
                      n_heads=args["num_heads"],
                      n_head_channels=args["num_head_channels"],
                      in_channels=in_channels)
    # small linear schedule (1000 time step , linear schaduler)
    betas = get_beta_schedule(args['T'],
                              args['beta_schedule'])
    diffusion = GaussianDiffusionModel(args['img_size'], #  [128, 128]
                                       betas,            #  1
                                       loss_weight=args['loss_weight'], # none
                                       loss_type=args['loss-type'],     # l2
                                       noise= args["noise_fn"],          # none
                                       img_channels=in_channels)        # 1
    if resume:
        if "unet" in resume:
            model.load_state_dict(resume["unet"])
        else:
            model.load_state_dict(resume["ema"])
        ema = UNetModel(args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'],
                        dropout=args["dropout"], n_heads=args["num_heads"], n_head_channels=args["num_head_channels"],
                        in_channels=in_channels)
        ema.load_state_dict(resume["ema"])
        start_epoch = resume['n_epoch']
    else:
        start_epoch = 0
        ema = copy.deepcopy(model)
    tqdm_epoch = range(start_epoch, args['EPOCHS'] + 1)
    model.to(device)
    ema.to(device)

    print(f'\n step 7. optimizer')
    optimiser = optim.AdamW(model.parameters(),
                            lr=args['lr'],
                            weight_decay=args['weight_decay'], betas=(0.9, 0.999))
    if resume:
        optimiser.load_state_dict(resume["optimizer_state_dict"])
    del resume

    print(f'\n step 8. training')
    start_time = time.time()
    losses = []
    vlb = collections.deque([], maxlen=10)
    iters = range(100 // args['Batch_Size']) if args["dataset"].lower() != "cifar" else range(200)
    for epoch in tqdm_epoch:
        mean_loss = []
        progress_bar = tqdm(enumerate(training_dataset_loader), total=len(training_dataset_loader), ncols=70)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, data in progress_bar:
            if args["dataset"] == "cifar":
                x = data[0].to(device)
            else:
                x = data["image"]
                x = x.to(device) # batch, channel, w, h
            # GaussianDiffusionModel.p_loss
            loss, estimates = diffusion.p_loss(model, x, args)
            noisy, est = estimates[1], estimates[2]
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimiser.step()
            update_ema_params(ema, model)
            mean_loss.append(loss.data.cpu())
            if epoch % 50 == 0 and step == 0:
                row_size = min(8, args['Batch_Size'])
                training_outputs(diffusion, x, est, noisy, epoch, row_size,
                                 save_imgs=args['save_imgs'], # true
                                 save_vids=args['save_vids'], # true
                                 ema=ema, args=args)





        losses.append(np.mean(mean_loss))
        if epoch % 200 == 0:
            time_taken = time.time() - start_time
            remaining_epochs = args['EPOCHS'] - epoch
            time_per_epoch = time_taken / (epoch + 1 - start_epoch)
            hours = remaining_epochs * time_per_epoch / 3600
            mins = (hours % 1) * 60
            hours = int(hours)
            vlb_terms = diffusion.calc_total_vlb(x, model, args)
            vlb.append(vlb_terms["total_vlb"].mean(dim=-1).cpu().item())
            print(
                f"epoch: {epoch}, most recent total VLB: {vlb[-1]} mean total VLB:"
                f" {np.mean(vlb):.4f}, "
                f"prior vlb: {vlb_terms['prior_vlb'].mean(dim=-1).cpu().item():.2f}, vb: "
                f"{torch.mean(vlb_terms['vb'], dim=list(range(2))).cpu().item():.2f}, x_0_mse: "
                f"{torch.mean(vlb_terms['x_0_mse'], dim=list(range(2))).cpu().item():.2f}, mse: "
                f"{torch.mean(vlb_terms['mse'], dim=list(range(2))).cpu().item():.2f}"
                f" time elapsed {int(time_taken / 3600)}:{((time_taken / 3600) % 1) * 60:02.0f}, "
                f"est time remaining: {hours}:{mins:02.0f}\r")            
    #    if epoch % 1000 == 0 and epoch >= 0:
    #        save(unet=model, args=args, optimiser=optimiser, final=False, ema=ema, epoch=epoch)
    #save(unet=model, args=args, optimiser=optimiser, final=True, ema=ema)
    #evaluation.testing(testing_dataset_loader, diffusion, ema=ema, args=args, model=model)

    # 
    # if resuming, loaded model is attached to the dictionary

    # load, pass args
    # remove checkpoints after final_param is saved (due to storage requirements)
    #for file_remove in os.listdir(f'./model/diff-params-ARGS={args["arg_num"]}/checkpoint'):
    #    os.remove(os.path.join(f'./model/diff-params-ARGS={args["arg_num"]}/checkpoint', file_remove))
    #os.removedirs(f'./model/diff-params-ARGS={args["arg_num"]}/checkpoint')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', type=str,
                        default = f'/data7/sooyeon/medical_image/anoddpm_result/20231119_dental_test')
    parser_argument = vars(parser.parse_args())

    file_dir = f'/data7/sooyeon/medical_image/anoddpm_result/test_args/args11.json'
    # sys.argv = ['C:\\Users\\hpuser\\PycharmProjects\\medical_image\\AnoDDPM\\diffusion_training.py']
    sys.argv.append(file_dir)
    files = sys.argv[1:]
    resume = 0
    file = files[0]
    with open(file, 'r') as f:
        args = json.load(f)
    args = defaultdict_from_json(args)
    args['resume'] = resume
    args.update(parser_argument)
    main(args)

