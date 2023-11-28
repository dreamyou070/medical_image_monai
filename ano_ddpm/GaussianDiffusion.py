from __future__ import annotations
import random
import matplotlib.pyplot as plt
import numpy as np
import evaluation
from helpers import *
from simplex import Simplex_CLASS

def get_beta_schedule(num_diffusion_steps, name="cosine"):
    betas = []
    if name == "cosine":
        max_beta = 0.999
        f = lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
        for i in range(num_diffusion_steps):
            t1 = i / num_diffusion_steps
            t2 = (i + 1) / num_diffusion_steps
            betas.append(min(1 - f(t2) / f(t1), max_beta))
        betas = np.array(betas)
    elif name == "linear":
        scale = 1000 / num_diffusion_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        betas = np.linspace(beta_start, beta_end, num_diffusion_steps, dtype=np.float64)

    elif name == 'sinusoidal' :
        # want green should be zero
        # that means alphas be zero
        f = lambda t: (np.pi / 2) * ((t / num_diffusion_steps) + 0.0008 / 1 + 0.0008)
        for i in range(num_diffusion_steps):
            value = np.sin(f(i)) * 0.02
            betas.append(value)
        betas = np.array(betas)

    else:
        raise NotImplementedError(f"unknown beta schedule: {name}")
    return betas

def extract(arr, timesteps, broadcast_shape, device):
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape).to(device)

def mean_flat(tensor):
    return torch.mean(tensor,
                      dim=list(range(1, len(tensor.shape))))

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL Divergence between two gaussians
    :param mean1:
    :param logvar1:
    :param mean2:
    :param logvar2:
    :return: KL Divergence between N(mean1,logvar1^2) & N(mean2,logvar2^2))
    """
    return 0.5 * (-1 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """

    a = torch.Tensor([np.sqrt(2.0 / np.pi)]).to(x.device)
    b = (x + 0.044715 * torch.pow(x, 3))
    return 0.5 * (1.0 + torch.tanh(a * b))


def discretised_gaussian_log_likelihood(x, means, log_scales):

    assert x.shape == means.shape == log_scales.shape

    centered_x = x - means

    inv_stdv = torch.exp(-log_scales)

    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))

    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))

    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(x < -0.999, log_cdf_plus,
                            torch.where(x > 0.999, log_one_minus_cdf_min,
                                        torch.log(cdf_delta.clamp(min=1e-12))),)
    assert log_probs.shape == x.shape
    return log_probs


def generate_simplex_noise(Simplex_instance, x, t,
                           random_param=False,
                           octave=6,
                           persistence=0.8,
                           frequency=64,
                           in_channels=1):

    noise = torch.empty(x.shape).to(x.device)
    for i in range(in_channels):
        Simplex_instance.newSeed()
        if random_param:
            param = random.choice(
                [(2, 0.6, 16), (6, 0.6, 32), (7, 0.7, 32), (10, 0.8, 64), (5, 0.8, 16), (4, 0.6, 16), (1, 0.6, 64),
                 (7, 0.8, 128), (6, 0.9, 64), (2, 0.85, 128), (2, 0.85, 64), (2, 0.85, 32), (2, 0.85, 16),
                 (2, 0.85, 8),
                 (2, 0.85, 4), (2, 0.85, 2), (1, 0.85, 128), (1, 0.85, 64), (1, 0.85, 32), (1, 0.85, 16),
                 (1, 0.85, 8),
                 (1, 0.85, 4), (1, 0.85, 2), ]
            )
            # 2D octaves seem to introduce directional artifacts in the top left
            noise[:, i, ...] = torch.unsqueeze(
                torch.from_numpy(
                    # Simplex_instance.rand_2d_octaves(
                    #         x.shape[-2:], param[0], param[1],
                    #         param[2]
                    #         )
                    Simplex_instance.rand_3d_fixed_T_octaves(
                        x.shape[-2:], t.detach().cpu().numpy(), param[0], param[1],
                        param[2]
                    )
                ).to(x.device), 0
            ).repeat(x.shape[0], 1, 1, 1)

        # -----------------------------------------------------------------------------------------------------------
        # timewise noise ???
        # Batch size,
        #noise_1 = Simplex_instance.rand_3d_fixed_T_octaves(x.shape[-2:],
        #                                                   torch.Tensor([1000]).cpu().numpy(),
        #                                                   octave,
        #                                                   persistence,
        #                                                   frequency)
        noise_1 = Simplex_instance.rand_2d_octaves(x.shape[-2:],
                                                   octave,
                                                   persistence,
                                                   frequency)

        #noise_1 = Simplex_instance.rand_3d_fixed_T_octaves(x.shape[-2:],
        #                                                   t.detach().cpu().numpy(),
        #                                                   octave,
        #                                                   persistence,
        #                                                   frequency)
        torch_noise = torch.from_numpy(noise_1).to(x.device).squeeze()
        #print(f"torch_noise shape (1, 256,256) : {torch_noise.shape}")
        batch_torch_noise = torch_noise.repeat(x.shape[0], 1, 1)
        noise[:, i, ...] = torch_noise
    return noise

class DDPMVarianceType():
    """
    Valid names for DDPM Scheduler's `variance_type` argument. Options to clip the variance used when adding noise
    to the denoised sample.
    """
    FIXED_SMALL = "fixed_small"
    FIXED_LARGE = "fixed_large"
    LEARNED = "learned"
    LEARNED_RANGE = "learned_range"

def random_noise(Simplex_instance, x, t):
    param = random.choice(["gauss", "simplex"])
    if param == "gauss":
        return torch.randn_like(x)
    else:
        return generate_simplex_noise(Simplex_instance, x, t)

class DDPMPredictionType():
    EPSILON = "epsilon"
    SAMPLE = "sample"
    V_PREDICTION = "v_prediction"

class GaussianDiffusionModel:

    def __init__(self,
                 img_size,
                 betas,
                 img_channels=1,
                 loss_type="l2",  # l2,l1 hybrid
                 loss_weight='none',  # prop t / uniform / None
                 noise="gauss",  # gauss / perlin / simplex
                 clip_sample: bool = True,):
        super().__init__()

        if noise == "gauss":
            self.noise_fn = lambda x, t: torch.randn_like(x)
        else:
            self.simplex = Simplex_CLASS()
            if noise == "simplex_randParam":
                self.noise_fn = lambda x, t: generate_simplex_noise(self.simplex, x, t, True, octave=6, frequency=64, in_channels=img_channels)
            elif noise == "random":
                self.noise_fn = lambda x, t: random_noise(self.simplex, x, t)
            else: # simplex
                self.noise_fn = lambda x, t, octave, frequency, : generate_simplex_noise(self.simplex, x, t, False, octave=octave,
                                                                                         frequency = frequency, in_channels=img_channels) # 1
        self.img_size = img_size
        self.img_channels = img_channels
        self.loss_type = loss_type
        self.num_timesteps = len(betas)
        if loss_weight == 'prop-t':
            self.weights = np.arange(self.num_timesteps, 0, -1)
        elif loss_weight == "uniform":
            self.weights = np.ones(self.num_timesteps)
        self.loss_weight = loss_weight
        alphas = 1 - betas
        self.alphas = alphas
        self.betas = betas
        self.sqrt_alphas = np.sqrt(alphas)
        self.sqrt_betas = np.sqrt(betas)

        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        self.posterior_mean_coef1 = ( betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod))

        self.prediction_type = DDPMPredictionType.EPSILON
        self.variance_type = DDPMVarianceType.FIXED_LARGE
        self.clip_sample = clip_sample

    def check_noise(self, args, x_0):
        t = torch.randint(0, min(args.sample_distance, self.num_timesteps), (x_0.shape[0],), device=x_0.device)
        noise = self.noise_fn(x_0, t).float()
        first_noise = noise[0, ...]
        return first_noise

    def _get_variance(self, timestep: int, predicted_variance: torch.Tensor | None = None) -> torch.Tensor:
        """ original paper, (7) equation,  the variance of the posterior at timestep t.
        Args:
            timestep: current timestep.
            predicted_variance: variance predicted by the model.
        Returns:
            Returns the variance """
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[timestep - 1] if timestep > 0 else self.one
        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * self.betas[timestep]
        # hacks - were probably added for training stability
        if self.variance_type == DDPMVarianceType.FIXED_SMALL:
            variance = torch.clamp(variance, min=1e-20)
        elif self.variance_type == DDPMVarianceType.FIXED_LARGE:
            variance = self.betas[timestep]
        elif self.variance_type == DDPMVarianceType.LEARNED:
            return predicted_variance
        elif self.variance_type == DDPMVarianceType.LEARNED_RANGE:
            min_log = variance
            max_log = self.betas[timestep]
            frac = (predicted_variance + 1) / 2
            variance = frac * max_log + (1 - frac) * min_log
        return variance

    def p_loss(self, model, x_0, args):
        """ calculate total loss """
        if self.loss_weight == "none":
            if args.train_start:
                t = torch.randint(0, min(args.sample_distance, self.num_timesteps), (x_0.shape[0],),
                                  device=x_0.device)
            else:
                t = torch.randint(0, self.num_timesteps, (x_0.shape[0],), device=x_0.device)
            weights = 1
        else:
            t, weights = self.sample_t_with_weights(x_0.shape[0], x_0.device)
        loss, x_t, eps_t = self.calc_loss(model, x_0, t)
        loss = ((loss["loss"] * weights).mean(),
                (loss, x_t, eps_t))
        return loss

    def calc_loss(self, model, x_0, t):
        noise = self.noise_fn(x_0, t).float()
        x_t = self.sample_q(x_0, t, noise)
        estimate_noise = model(x_t, t)
        loss = {}
        if self.loss_type == "l1":
            loss["loss"] = mean_flat((estimate_noise - noise).abs())
        elif self.loss_type == "l2":
            loss["loss"] = mean_flat((estimate_noise - noise).square())
        elif self.loss_type == "hybrid":
            # add vlb term
            loss["vlb"] = self.calc_vlb_xt(model, x_0, x_t, t, estimate_noise)["output"]
            loss["loss"] = loss["vlb"] + mean_flat((estimate_noise - noise).square())
        else:
            loss["loss"] = mean_flat((estimate_noise - noise).square())
        return loss, x_t, estimate_noise

    # -----------------------------------------------------------------------------------------------------------------
    def sample_q(self, x_0, t, noise):
        """ add noise , equation (4)  : q (x_t | x_0 )
            :param x_0, :param t, :param noise,  :return:
            self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        """
        return (extract(self.sqrt_alphas_cumprod, t, x_0.shape, x_0.device) * x_0 +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape, x_0.device) * noise)

    def q_mean_variance(self, x_0, t):
        mean = (extract(self.sqrt_alphas_cumprod, t, x_0.shape, x_0.device) * x_0 )
        variance = extract(1.0 - self.alphas_cumprod, t, x_0.shape, x_0.device)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_0.shape, x_0.device)
        return mean, variance, log_variance

    def q_posterior_mean_variance(self, x_0, x_t, t):
        """  Original Paper, equation (7)
        Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0) """
        # mu (x_t,x_0) = \frac{\sqrt{alphacumprod prev} betas}{1-alphacumprod} *x_0
        # + \frac{\sqrt{alphas}(1-alphacumprod prev)}{ 1- alphacumprod} * x_t
        posterior_mean = (extract(self.posterior_mean_coef1, t, x_t.shape, x_t.device) * x_0
                        + extract(self.posterior_mean_coef2, t, x_t.shape, x_t.device) * x_t)
        # var = \frac{1-alphacumprod prev}{1-alphacumprod} * betas
        posterior_var = extract(self.posterior_variance, t, x_t.shape, x_t.device)
        posterior_log_var_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape, x_t.device)
        return posterior_mean, posterior_var, posterior_log_var_clipped
    def sample_t_with_weights(self, b_size, device):
        p = self.weights / np.sum(self.weights)
        indices_np = np.random.choice(len(p), size=b_size, p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / len(p) * p[indices_np]
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights
    def predict_x_0_from_eps(self, x_t, t, eps):
        """ from equation (4), just equation trimming, eps is predicted noise from x_t by model """
        return (extract(self.sqrt_recip_alphas_cumprod,  t, x_t.shape, x_t.device) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod,t, x_t.shape, x_t.device) * eps)
    def predict_eps_from_x_0(self, x_t, t, pred_x_0):
        """ from equation (4), just equation trimming -> this is not using model but just scheduler """
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape, x_t.device) * x_t - pred_x_0) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, x_t.device)
    def p_mean_variance(self, model, x_t, t, estimate_noise=None):
        # one step stepping #
        """ Finds the mean & variance from N(x_{t-1}; mu_theta(x_t,t), sigma_theta (x_t,t)) : equation (1) """
        if estimate_noise == None:
            estimate_noise = model(x_t, t)
        model_var = np.append(self.posterior_variance[1], self.betas[1:])
        model_logvar = np.log(model_var)
        model_var = extract(model_var, t, x_t.shape, x_t.device)
        model_logvar = extract(model_logvar, t, x_t.shape, x_t.device)
        pred_x_0 = self.predict_x_0_from_eps(x_t,t,estimate_noise).clamp(-1, 1)
        model_mean, _, _ = self.q_posterior_mean_variance(pred_x_0, x_t, t)
        return {"mean":model_mean, "variance":model_var, "log_variance": model_logvar,"pred_x_0":pred_x_0,}


    def sample_p(self, model, x_t, t, denoise_fn="gauss"):
        """ equation (1) """
        out = self.p_mean_variance(model, x_t, t)
        if type(denoise_fn) == str:
            if denoise_fn == "gauss":
                noise = torch.randn_like(x_t)
            elif denoise_fn == "noise_fn":
                noise = self.noise_fn(x_t, t).float()
            elif denoise_fn == "random":
                noise = torch.randn_like(x_t)
            else:
                noise = generate_simplex_noise(self.simplex, x_t, t, False, in_channels=self.img_channels).float()
        else:
            noise = denoise_fn(x_t, t)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_x_0": out["pred_x_0"]}

    def step(self, model, noise_pred, x_t, t, denoise_fn="gauss"):
        out = self.p_mean_variance(model, x_t, t)
        sample = out["mean"] + torch.exp(0.5 * out["log_variance"]) * noise_pred
        return sample

    def dental_forward_backward(self,
                                model,
                                x,
                                args,
                                device,
                                t_distance=None, ):

        # -------------------------------------------------------------------------------------------------------------
        # 0) set t
        if t_distance is None:
            t_distance = self.num_timesteps
        t_tensor = torch.tensor([t_distance - 1], device=x.device).repeat(x.shape[0])

        # -------------------------------------------------------------------------------------------------------------
        # 1) generate noise
        if args.use_simplex_noise:
            noise = self.noise_fn(x=x, t=t_tensor, octave=6, frequency=64).float()
        else:
            noise = torch.rand_like(x).float().to(device)
            #noise = torch.rand_like(x).float().to(device)

        # -------------------------------------------------------------------------------------------------------------
        # 2) noisy x
        x = self.sample_q(x,t_tensor,noise).float()

        # -------------------------------------------------------------------------------------------------------------
        # 3) generating
        for t in range(int(t_distance) - 1, -1, -1):
            print('t : ', t)
            t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
            with torch.no_grad():
                model_output = model(x,t_tensor)
                # 2. compute previous image: x_t -> x_t-1
                x, _ = self.step(model_output, t, x)
                first_sample = x[0]
                print(f'first_sample shape : {first_sample.shape}')
                # save intermediate result
                import torchvision.transforms as torch_transforms
                torch_transforms.ToPILImage()(first_sample.squeeze()).save(f'intermediate_{t}.png')
        return x.detach()

    # -----------------------------------------------------------------------------------------------------------------
    def forward_backward(self,
                         model,
                         x, see_whole_sequence="half", t_distance=None, denoise_fn="gauss",):
        assert see_whole_sequence == "whole" or see_whole_sequence == "half" or see_whole_sequence == None

        if t_distance == 0:
            return x.detach()

        if t_distance is None:
            t_distance = self.num_timesteps
        seq = [x.cpu().detach()]
        if see_whole_sequence == "whole":
            for t in range(int(t_distance)):
                t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
                # noise = torch.randn_like(x)
                noise = self.noise_fn(x, t_batch).float()
                with torch.no_grad():
                    x = self.sample_q_gradual(x, t_batch, noise)

                seq.append(x.cpu().detach())


        else:
            # x = self.sample_q(x,torch.tensor([t_distance], device=x.device).repeat(x.shape[0]),torch.randn_like(x))
            t_tensor = torch.tensor([t_distance - 1], device=x.device).repeat(x.shape[0])
            # ---------------------------------------------------------------------------------------------------------
            # def sample_q = make noise latent
            x = self.sample_q(x, t_tensor, self.noise_fn(x, t_tensor).float())
            if see_whole_sequence == "half":
                seq.append(x.cpu().detach())

        for t in range(int(t_distance) - 1, -1, -1):
            t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
            with torch.no_grad():
                # sample_p =
                out = self.sample_p(model, x, t_batch, denoise_fn)
                x = out["sample"]
            if see_whole_sequence:
                seq.append(x.cpu().detach())

        return x.detach() if not see_whole_sequence else seq

    def sample_q_gradual(self, x_t, t, noise):
        """
        q (x_t | x_{t-1})
        :param x_t:
        :param t:
        :param noise:
        :return:
        """
        return (extract(self.sqrt_alphas, t, x_t.shape, x_t.device) * x_t +
                extract(self.sqrt_betas, t, x_t.shape, x_t.device) * noise)

    def prior_vlb(self, x_0, args):

        # --------------------------------------------------------------------------------------------------------------
        # 1) calculate q (x_T | x_0) : final step mean, and log_variance
        t = torch.tensor([self.num_timesteps - 1] * x_0.shape[0], device=x_0.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_0, t)

        # --------------------------------------------------------------------------------------------------------------
        # KL divergence between two probability
        kl_prior = normal_kl(mean1=qt_mean,                              logvar1=qt_log_variance,
                             mean2=torch.tensor(0.0, device=x_0.device), logvar2=torch.tensor(0.0, device=x_0.device))
        return mean_flat(kl_prior) / np.log(2.0)

    def prior_vlb(self, x_0, args):
        t = torch.tensor([self.num_timesteps - 1] * x_0.shape[0], device=x_0.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_0, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=torch.tensor(0.0, device=x_0.device),
                             logvar2=torch.tensor(0.0, device=x_0.device))
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_vlb_xt(self, model, x_0, x_t, t, estimate_noise=None):
        timestep = int(t[0])
        t_batch_patch = (torch.ones_like(x_0) * timestep).to(x_0.device)

        # --------------------------------------------------------------------------------------------------------------
        # 1) compare scheduling one step reverse and model one step reverse
        true_mean, _, true_log_var = self.q_posterior_mean_variance(x_0, x_t, t)


        output = self.p_mean_variance(model, x_t, t, estimate_noise)
        model_mean = output["mean"]
        model_log_var = output["log_variance"]

        whole_kl = normal_kl(true_mean, true_log_var, model_mean, model_log_var)
        kl = mean_flat(whole_kl) / np.log(2.0)







        # --------------------------------------------------------------------------------------------------------------
        # if timestep is 0, it compare with x_0
        # independent discrete decoder (log likelihood)
        decoder_nll_ = -1 * discretised_gaussian_log_likelihood(x_0,output["mean"],
                                                                log_scales=0.5 * output["log_variance"])
        decoder_nll = mean_flat(decoder_nll_) / np.log(2.0)



        # --------------------------------------------------------------------------------------------------------------
        # 3) if t == 0 : the value is decoder negative log likelihood
        #    else : kl between schedule value and model value
        patch_nll = torch.where((t_batch_patch == 0), decoder_nll_, whole_kl)
        nll = torch.where((t == 0), decoder_nll, kl)
        return {"output": nll,
                "pred_x_0": output["pred_x_0"],
                #"whole_kl" : whole_kl,
                "whole_kl": patch_nll
                }

    def calc_total_vlb_in_sample_distance(self, x_0, model, args):
        sample_distance = args.sample_distance
        vb, vb_whole = [], []
        x_0_mse = []
        noise_mse = []
        for t in reversed(list(range(sample_distance))):
            # from 1000 to 0  (that means generating)
            t_batch = torch.tensor([t] * x_0.shape[0], device=x_0.device)
            noise = torch.randn_like(x_0)
            x_t = self.sample_q(x_0=x_0, t=t_batch, noise=noise)
            # ----------------------------------------------------------------------------------------------------------
            # 1) Calculate VLB term at the current timestep
            with torch.no_grad():
                # when t != 0 : original kl divergence
                # when t == 0 : deconer negative log likelihood
                out = self.calc_vlb_xt(model, x_0=x_0, x_t=x_t, t=t_batch, )
                kl_divergence = out["output"]
                whole_kl = out["whole_kl"]
            vb.append(kl_divergence)
            vb_whole.append(whole_kl)
            # ----------------------------------------------------------------------------------------------------------
            # 2) every timestep, MSE between model_x0 and true_x)
            model_x0 = out['pred_x_0']
            true_x0 = x_0
            x_0_mse.append(mean_flat((model_x0 - true_x0) ** 2))
            # ----------------------------------------------------------------------------------------------------------
            # 3) every timestep, MSE between model_noise and true_noise
            model_noise = self.predict_eps_from_x_0(x_t, t_batch, out["pred_x_0"])
            true_noise = noise
            noise_mse.append(mean_flat((model_noise - true_noise) ** 2))
        # vb = [Batch, number of timestps = 1000]
        whole_vb = torch.stack(vb_whole, dim=1)  # [batch, 1000, 1, W, H]
        vb = torch.stack(vb, dim=1)  # [batch, 1000]
        x_0_mse = torch.stack(x_0_mse, dim=1)
        noise_mse = torch.stack(noise_mse, dim=1)
        prior_vlb = self.prior_vlb(x_0, args)  # [batch]
        total_vlb = vb.sum(dim=1) + prior_vlb  # [batch]
        return {"total_vlb": total_vlb,
                "prior_vlb": prior_vlb,
                "vb": vb,
                "whole_vb": whole_vb,
                "x_0_mse": x_0_mse,
                "mse": noise_mse, }

    def calc_total_vlb(self, x_0, model, args):
        vb, vb_whole = [], []
        x_0_mse = []
        noise_mse = []
        for t in reversed(list(range(self.num_timesteps))):
            # from 1000 to 0  (that means generating)
            t_batch = torch.tensor([t] * x_0.shape[0], device=x_0.device)
            noise = torch.randn_like(x_0)
            x_t = self.sample_q(x_0=x_0, t=t_batch, noise=noise)
            # ----------------------------------------------------------------------------------------------------------
            # 1) Calculate VLB term at the current timestep
            with torch.no_grad():
                # when t != 0 : original kl divergence
                # when t == 0 : deconer negative log likelihood
                out = self.calc_vlb_xt(model,x_0=x_0,x_t=x_t,t=t_batch,)
                kl_divergence = out["output"]
                whole_kl = out["whole_kl"]
            vb.append(kl_divergence)
            vb_whole.append(whole_kl)
            # ----------------------------------------------------------------------------------------------------------
            # 2) every timestep, MSE between model_x0 and true_x)
            model_x0 = out['pred_x_0']
            true_x0 = x_0
            x_0_mse.append(mean_flat((model_x0 - true_x0) ** 2))
            # ----------------------------------------------------------------------------------------------------------
            # 3) every timestep, MSE between model_noise and true_noise
            model_noise = self.predict_eps_from_x_0(x_t, t_batch, out["pred_x_0"])
            true_noise = noise
            noise_mse.append(mean_flat((model_noise - true_noise) ** 2))
        # vb = [Batch, number of timestps = 1000]
        whole_vb = torch.stack(vb_whole, dim=1)  # [batch, 1000, 1, W, H]
        vb = torch.stack(vb, dim=1)        # [batch, 1000]
        x_0_mse = torch.stack(x_0_mse, dim=1)
        noise_mse = torch.stack(noise_mse, dim=1)
        prior_vlb = self.prior_vlb(x_0, args)     # [batch]
        total_vlb = vb.sum(dim=1) + prior_vlb     # [batch]
        return {"total_vlb": total_vlb,
                "prior_vlb": prior_vlb,
                "vb": vb,
                "whole_vb": whole_vb,
                "x_0_mse": x_0_mse,
                "mse": noise_mse,}

    def detection_A(self, model, x_0, args, file, mask, total_avg=2):
        for i in [f"./diffusion-videos/ARGS={args['arg_num']}/Anomalous/{file[0]}",
                  f"./diffusion-videos/ARGS={args['arg_num']}/Anomalous/{file[0]}/{file[1]}/",
                  f"./diffusion-videos/ARGS={args['arg_num']}/Anomalous/{file[0]}/{file[1]}/A"]:
            try:
                os.makedirs(i)
            except OSError:
                pass

        for i in range(7, 0, -1):
            freq = 2 ** i
            self.noise_fn = lambda x, t: generate_simplex_noise(
                self.simplex, x, t, False, frequency=freq,
                in_channels=self.img_channels
            )

            for t_distance in range(50, int(args.T * 0.6), 50):
                output = torch.empty((total_avg, 1, *args.img_size), device=x_0.device)
                for avg in range(total_avg):
                    t_tensor = torch.tensor([t_distance], device=x_0.device).repeat(x_0.shape[0])
                    x = self.sample_q(x_0, t_tensor,self.noise_fn(x_0, t_tensor).float())
                    for t in range(int(t_distance) - 1, -1, -1):
                        t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
                        with torch.no_grad():
                            out = self.sample_p(model, x, t_batch)
                            # ------------------------------------------------------------------------------------------
                            # one step forward
                            x = out["sample"]
                    output[avg, ...] = x

                # save image containing initial, each final denoised image, mean & mse
                output_mean = torch.mean(output, dim=0).reshape(1, 1, *args["img_size"])
                mse = ((output_mean - x_0).square() * 2) - 1
                mse_threshold = mse > 0
                mse_threshold = (mse_threshold.float() * 2) - 1
                out = torch.cat([x_0, output[:3], output_mean, mse, mse_threshold, mask])

                temp = os.listdir(f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{file[0]}/{file[1]}/A')

                plt.imshow(gridify_output(out, 4), cmap='gray')
                plt.axis('off')
                plt.savefig(
                    f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{file[0]}/{file[1]}/A/freq={i}-t'
                    f'={t_distance}-{len(temp) + 1}.png'
                )
                plt.clf()

    def detection_B(self, model, x_0, args, file, mask, denoise_fn="gauss", total_avg=5):
        assert type(file) == tuple
        for i in [f"./diffusion-videos/ARGS={args['arg_num']}/Anomalous/{file[0]}",
                  f"./diffusion-videos/ARGS={args['arg_num']}/Anomalous/{file[0]}/{file[1]}",
                  f"./diffusion-videos/ARGS={args['arg_num']}/Anomalous/{file[0]}/{file[1]}/{denoise_fn}"]:
            try:
                os.makedirs(i)
            except OSError:
                pass
        if denoise_fn == "octave":
            end = int(args["T"] * 0.6)
            self.noise_fn = lambda x, t: generate_simplex_noise(
                self.simplex, x, t, False, frequency=64, octave=6,
                persistence=0.8
            ).float()
        else:
            end = int(args["T"] * 0.8)
            self.noise_fn = lambda x, t: torch.randn_like(x)
        # multiprocessing?
        dice_coeff = []
        for t_distance in range(50, end, 50):
            output = torch.empty((total_avg, 1, *args["img_size"]), device=x_0.device)
            for avg in range(total_avg):

                t_tensor = torch.tensor([t_distance], device=x_0.device).repeat(x_0.shape[0])
                x = self.sample_q(
                    x_0, t_tensor,
                    self.noise_fn(x_0, t_tensor).float()
                )

                for t in range(int(t_distance) - 1, -1, -1):
                    t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
                    with torch.no_grad():
                        out = self.sample_p(model, x, t_batch)
                        x = out["sample"]

                output[avg, ...] = x

            # save image containing initial, each final denoised image, mean & mse
            output_mean = torch.mean(output, dim=[0]).reshape(1, 1, *args["img_size"])

            temp = os.listdir(f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{file[0]}/{file[1]}/{denoise_fn}')

            dice = evaluation.heatmap(
                real=x_0, recon=output_mean, mask=mask,
                filename=f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{file[0]}/{file[1]}/'
                         f'{denoise_fn}/heatmap-t={t_distance}-{len(temp) + 1}.png'
            )

            mse = ((output_mean - x_0).square() * 2) - 1
            mse_threshold = mse > 0
            mse_threshold = (mse_threshold.float() * 2) - 1
            out = torch.cat([x_0, output[:3], output_mean, mse, mse_threshold, mask])

            plt.imshow(gridify_output(out, 4), cmap='gray')
            plt.axis('off')
            plt.savefig(
                f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{file[0]}/{file[1]}/{denoise_fn}/t'
                f'={t_distance}-{len(temp) + 1}.png'
            )
            plt.clf()

            dice_coeff.append(dice)
        return dice_coeff

    def detection_A_fixedT(self, model, x_0, args, mask, end_freq=6):
        t_distance = 250

        output = torch.empty((6 * end_freq, 1, *args["img_size"]), device=x_0.device)
        for i in range(1, end_freq + 1):

            freq = 2 ** i
            noise_fn = lambda x, t: generate_simplex_noise(self.simplex, x, t, False, frequency=freq).float()

            t_tensor = torch.tensor([t_distance - 1], device=x_0.device).repeat(x_0.shape[0])
            x = self.sample_q(
                x_0, t_tensor,
                noise_fn(x_0, t_tensor).float()
            )
            x_noised = x.clone().detach()
            for t in range(int(t_distance) - 1, -1, -1):
                t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
                with torch.no_grad():
                    out = self.sample_p(model, x, t_batch, denoise_fn=noise_fn)
                    x = out["sample"]

            mse = ((x_0 - x).square() * 2) - 1
            mse_threshold = mse > 0
            mse_threshold = (mse_threshold.float() * 2) - 1

            output[(i - 1) * 6:i * 6, ...] = torch.cat((x_0, x_noised, x, mse, mse_threshold, mask))

        return output

    # -----------------------------------------------------------------------------------------------------------------

x = """
Two methods of detection:
A - using varying simplex frequencies
B - using octave based simplex noise
C - gaussian based (same as B but gaussian)

A: for i in range(6,0,-1):
    2**i == frequency
   Frequency = 64: Sample 10 times at t=50, denoise and average
   Repeat at t = range (50, ARGS["sample distance"], 50)   
   Note simplex noise is fixed frequency ie no octave mixture   
B: Using some initial "good" simplex octave parameters such as 64 freq, oct = 6, persistence= 0.9   
   Sample 10 times at t=50, denoise and average
   Repeat at t = range (50, ARGS["sample distance"], 50)   
"""
