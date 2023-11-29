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
    model, diffusion = create_model_and_diffusion(**args_to_dict(args,
                                                                 model_and_diffusion_defaults(use_kl = True).keys()))
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # 1) base argument
    # data_dir
    defaults = dict(data_dir="/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_256/train/original",
                    schedule_sampler="uniform",
                    lr=1e-4,
                    weight_decay=0.0,
                    lr_anneal_steps=0,
                    batch_size=1,
                    microbatch=-1,  # -1 disables microbatches
                    ema_rate="0.9999",  # comma-separated list of EMA values
                    log_interval=10,
                    save_interval=10000,
                    resume_checkpoint="",
                    use_fp16=False,
                    fp16_scale_growth=1e-3,)
    # 2) model diffusion argument
    defaults.update(model_and_diffusion_defaults(use_kl = True))
    add_dict_to_argparser(parser, defaults)
    args = parser.parse_args()
    main(args)
