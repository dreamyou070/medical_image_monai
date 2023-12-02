import argparse, os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improved_diffusion.resample import UniformSampler,LossSecondMomentResampler
from improved_diffusion.script_util import (create_model, create_gaussian_diffusion, )
from improved_diffusion.image_datasets import ImageDataset, DataLoader
from improved_diffusion.train_util import TrainLoop
from mpi4py import MPI
import torchvision.transforms as torch_transforms
import copy
import functools
import os
import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from improved_diffusion import dist_util, logger
from improved_diffusion.fp16_util import (make_master_params,master_params_to_model_params,model_grads_to_master_grads,
                                          unflatten_master_params,zero_grad,)
from improved_diffusion.nn import update_ema
from improved_diffusion.resample import LossAwareSampler, UniformSampler
from improved_diffusion.train_util import (find_resume_checkpoint, find_ema_checkpoint, log_loss_dict, make_master_params,
                                           master_params_to_model_params, model_grads_to_master_grads, update_ema,
                                           parse_resume_step_from_filename, get_blob_logdir)

INITIAL_LOG_LOSS_SCALE = 20.0

class TrainLoop:
    def __init__(self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        inference_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):

        self.model = model
        # diffusion = Space
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = ([ema_rate] if isinstance(ema_rate, float) else [float(x) for x in ema_rate.split(",")])
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.inference_interval = inference_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth

        # (1) scheduler
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)

        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()
        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()
        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()
        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            self.ema_params = [self._load_ema_parameters(rate) for rate in self.ema_rate]
        else:
            self.ema_params = [copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))]

        if th.cuda.is_available():
            self.use_ddp = True
            #self.ddp_model = DDP(self.model,
                                 #device_ids=[dist_util.dev()], output_device=dist_util.dev(),
            #                     broadcast_buffers=False,
            #                     bucket_cap_mb=128,
            #                     find_unused_parameters=False,)
            self.ddp_model = self.model
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(resume_checkpoint, map_location=dist_util.dev()))
        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(ema_checkpoint, map_location=dist_util.dev())
                ema_params = self._state_dict_to_master_params(state_dict)
        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt")
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    @ th.no_grad()
    def inference(self, data):

        # (1) make random noisy sample
        with th.no_grad():
            random_int = th.randint(0, 1000, (1,)).item()
            for i in range(random_int, 0, -1):
                b_size = data.shape[0]
                t = th.Tensor([i]).repeat(b_size, ).long().to(args.device)
                output = self.diffusion.ddim_sample(model=self.ddp_model,
                                                    x=data.to(args.device),
                                                    t=t)
                data = output['sample']
                pred_x_0 = output["pred_xstart"]
                pil_data = torch_transforms.ToPILImage()(data.cpu().squeeze())
                pil_data.save(f"sample_{i}.png")
            final_sample = data
            pil_data = torch_transforms.ToPILImage()(final_sample.cpu().squeeze())
            pil_data.save(f"sample_{0}.png")

    def run_loop(self):

        while (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
            # 1) get data
            for batch, cond in self.data :
                # batch = np image, cond = dict type local class information
                self.run_step(batch, cond)
                if self.step % self.log_interval == 0:
                    logger.dumpkvs()
                if self.step % self.save_interval == 0:
                    self.save()
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                if self.step % self.inference_interval == 0 :
                    self.inference(batch)
                self.step += 1
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def forward_backward(self, batch, cond):
        zero_grad(self.model_params)
        batch_s = batch.shape[0]
        for i in range(0, batch.shape[0], self.microbatch):
            # (1) batch sample
            micro = batch[i: i + self.microbatch].to(args.device)
            # (2) condition sample
            #micro_cond = {k: v[i : i + self.microbatch].to(dist_util.dev())for k, v in cond.items()}
            micro_cond = {k: v[i: i + self.microbatch].to(args.device) for k, v in cond.items()}
            last_batch = (i + self.microbatch) >= batch.shape[0] # last_batch = True
            #t, weights = self.schedule_sampler.sample(micro.shape[0],dist_util.dev())
            # ----------------------------------------------------------------------------------------------------------
            # important timestep sampling
            # (1) sampling timestep (weights = just 1)
            t, weights = self.schedule_sampler.sample(micro.shape[0], args.device)
            # ----------------------------------------------------------------------------------------------------------
            # (2) compute losses : self.diffusion = SpacedDiffusion
            loss_fn = self.diffusion.training_losses
            compute_losses = functools.partial(loss_fn,                   # loss function
                                               self.ddp_model,           # model
                                               micro,                     # batch data
                                               t,                         # batch timestep
                                               model_kwargs=micro_cond,)  # {}
            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})
            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()


    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return
        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                save_dir = bf.join(args.experiment_dir, filename)
                with bf.BlobFile(save_dir, "wb") as f:
                    th.save(state_dict, f)
        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)
        if dist.get_rank() == 0:
            with bf.BlobFile(bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),"wb",) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                self.model.parameters(), master_params
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params

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
    logger.configure(dir = args.experiment_dir)

    print(f' step 2. creating model and diffusion')
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
                                          use_kl=args.use_kl, #######################################
                                          predict_xstart=args.predict_xstart,
                                          rescale_timesteps=args.rescale_timesteps,
                                          rescale_learned_sigmas=args.rescale_learned_sigmas, #######################################
                                          timestep_respacing=args.timestep_respacing,)
                                          #loss_type=args.loss_type,)
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
    dataset = ImageDataset(args.image_size,all_files,classes=classes,
                           shard=MPI.COMM_WORLD.Get_rank(),num_shards=MPI.COMM_WORLD.Get_size(), )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=True)
    #loader = yield from loader

    print(f' step 5. training...')
    trainer = TrainLoop(model=model,
                        diffusion=diffusion,
                        data=loader,
                        batch_size=args.batch_size,
                        microbatch=args.microbatch,
                        lr=args.lr,
                        ema_rate=args.ema_rate,
                        log_interval=args.log_interval,
                        save_interval=args.save_interval,
                        inference_interval=args.inference_interval,
                        resume_checkpoint=args.resume_checkpoint,
                        use_fp16=args.use_fp16,
                        fp16_scale_growth=args.fp16_scale_growth,
                        schedule_sampler=schedule_sampler,
                        weight_decay=args.weight_decay,
                        lr_anneal_steps=args.lr_anneal_steps,)
    trainer.run_loop()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument('--schedule_sampler', type=str, default='uniform')
    parser.add_argument('--experiment_dir', type=str,)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lr_anneal_steps', type=int, default=0)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--microbatch', type=int, default=-1)


    parser.add_argument('--ema_rate', type=str, default='0.9999')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--inference_interval', type=int, default=10)

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
