import argparse, os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import UniformSampler,LossSecondMomentResampler
from improved_diffusion.script_util import (model_and_diffusion_defaults,create_model_and_diffusion,args_to_dict,add_dict_to_argparser,
                                            create_model, create_gaussian_diffusion, )
from improved_diffusion.image_datasets import ImageDataset, DataLoader
from improved_diffusion.train_util import TrainLoop
import blobfile as bf
from mpi4py import MPI


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

def main(args):

    print(f' step 1. args: {args}')
    dist_util.setup_dist()
    logger.configure()

    print(f' step 2. creating model and diffusion...')
    model = create_model(args.image_size,
                         args.num_channels,
                         args.num_res_blocks,
                         learn_sigma=args.learn_sigma,
                         class_cond=args.class_cond,
                         use_checkpoint=args.use_checkpoint,
                         attention_resolutions=args.attention_resolutions,
                         num_heads=args.num_heads,
                         num_heads_upsample=args.num_heads_upsample,
                         use_scale_shift_norm=args.use_scale_shift_norm,
                         dropout=args.dropout,)
    diffusion = create_gaussian_diffusion(steps=args.diffusion_steps,
                                          learn_sigma=args.learn_sigma,
                                          sigma_small=args.sigma_small,
                                          noise_schedule=args.noise_schedule,
                                          use_kl=args.use_kl,
                                          predict_xstart=args.predict_xstart,
                                          rescale_timesteps=args.rescale_timesteps,
                                          rescale_learned_sigmas=args.rescale_learned_sigmas,
                                          timestep_respacing=args.timestep_respacing,)
    model.to(args.device)

    print(f' step 3. scheduler')
    # schedule_sampler = uniform
    # diffusion = diffusion model
    if args.schedule_sampler == 'uniform':
        schedule_sampler = UniformSampler(diffusion)
    else :
        schedule_sampler = LossSecondMomentResampler(diffusion)

    print(f' step 4. creating data loader...')
    all_files = _list_image_files_recursively(args.data_dir)
    classes = None
    if args.class_cond:
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(args.image_size,
                           all_files,
                           classes=classes,
                           shard=MPI.COMM_WORLD.Get_rank(),
                           num_shards=MPI.COMM_WORLD.Get_size(), )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=True)


    """
    

    

    

    print(f' step 5. training...')
    TrainLoop(model=model,
              diffusion=diffusion,
              data=data,
              batch_size=args.batch_size,
              microbatch=args.microbatch,
              lr=args.lr,
              ema_rate=args.ema_rate,
              log_interval=args.log_interval,
              save_interval=args.save_interval,
              resume_checkpoint=args.resume_checkpoint,
              use_fp16=args.use_fp16,
              fp16_scale_growth=args.fp16_scale_growth,
              schedule_sampler=schedule_sampler,
              weight_decay=args.weight_decay,
              lr_anneal_steps=args.lr_anneal_steps,).run_loop()
    """

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument('--scheduler_sampler', type=str, default='uniform')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lr_anneal_steps', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--microbatch', type=int, default=-1)
    parser.add_argument('--ema_rate', type=str, default='0.9999')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=10000)
    parser.add_argument('--resume_checkpoint', type=str, default='')
    parser.add_argument('--use_fp16', action='store_true')
    parser.add_argument('--fp16_scale_growth', type=float, default=1e-3)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_channels', type=int, default=128)
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_heads_upsample', type=int, default=-1)
    parser.add_argument('--attention_resolutions', type=str, default='16,8')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--learn_sigma', action='store_true')
    parser.add_argument('--sigma_small', action='store_true')
    parser.add_argument('--class_cond', action='store_true')
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--noise_schedule', type=str, default='linear')
    parser.add_argument('--timestep_respacing', type=str, default='')
    parser.add_argument('--use_kl', action='store_true')
    parser.add_argument('--predict_xstart', action='store_true')
    parser.add_argument('--rescale_timesteps', action='store_true')
    parser.add_argument('--rescale_learned_sigmas', action='store_true')
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.add_argument('--use_scale_shift_norm', action='store_true')
    parser.add_argument('--device', default = 'cuda:6')
    args = parser.parse_args()
    main(args)
