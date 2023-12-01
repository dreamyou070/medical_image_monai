import argparse, os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (model_and_diffusion_defaults,create_model_and_diffusion,args_to_dict,add_dict_to_argparser,)
from improved_diffusion.train_util import TrainLoop

def main(args):

    print(f' step 1. args: {args}')
    dist_util.setup_dist()
    logger.configure()

    print(f' step 2. creating model and diffusion...')
    # diffusion  = Space
    arg_dict = args_to_dict(args)
    model, diffusion = create_model_and_diffusion(arg_dict)
    """
    model.to(dist_util.dev())

    print(f' step 3. scheduler')
    # schedule_sampler = uniform
    # diffusion = diffusion model
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler,
                                                     diffusion)

    print(f' step 4. creating data loader...')
    data = load_data(data_dir=args.data_dir,
                     batch_size=args.batch_size,
                     image_size=args.image_size,
                     class_cond=args.class_cond,)

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
    args = parser.parse_args()
    main(args)
