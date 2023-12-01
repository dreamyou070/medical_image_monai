# https://github.com/openai/guided-diffusion/tree/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924
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
    else:
        raise NotImplementedError(f"unknown beta schedule: {name}")
    return betas


def extract(arr, timesteps, broadcast_shape, device):
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape).to(device)


def mean_flat(tensor):
    return torch.mean(tensor, dim=list(range(1, len(tensor.shape))))


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
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretised_gaussian_log_likelihood(x, means, log_scales):
    """
        Compute the log-likelihood of a Gaussian distribution discretizing to a
        given image.
        :param x: the target images. It is assumed that this was uint8 values,
                  rescaled to the range [-1, 1].
        :param means: the Gaussian mean Tensor.
        :param log_scales: the Gaussian log stddev Tensor.
        :return: a tensor like x of log probabilities (in nats).
        """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)

    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)

    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))

    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


def generate_simplex_noise(
        Simplex_instance, x, t, random_param=False, octave=6, persistence=0.8, frequency=64,
        in_channels=1
):
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
        # ------------------------------------------------------------------------------------
        simplex_noise = Simplex_instance.rand_3d_fixed_T_octaves(
            x.shape[-2:], t.detach().cpu().numpy(), octave,
            persistence, frequency)
        noise[:, i, ...] = torch.unsqueeze(torch.from_numpy(simplex_noise).to(x.device), 0).\
             repeat(x.shape[0], 1, 1, 1)
    return noise


def random_noise(Simplex_instance, x, t):
    param = random.choice(
        ["gauss", "simplex"]
    )
    if param == "gauss":
        return torch.randn_like(x)
    else:
        return generate_simplex_noise(Simplex_instance, x, t)


class GaussianDiffusionModel:

    def __init__(
            self,
            img_size,
            betas,
            img_channels=1,
            loss_type="l2",  # l2,l1 hybrid
            loss_weight='none',  # prop t / uniform / None
            noise="gauss",  # gauss / perlin / simplex
    ):
        super().__init__()
        if noise == "gauss":
            self.noise_fn = lambda x, t: torch.randn_like(x)

        else:
            self.simplex = Simplex_CLASS()
            if noise == "simplex_randParam":
                self.noise_fn = lambda x, t: generate_simplex_noise(self.simplex, x, t, True, in_channels=img_channels)
            elif noise == "random":
                self.noise_fn = lambda x, t: random_noise(self.simplex, x, t)
            else:
                self.noise_fn = lambda x, t: generate_simplex_noise(self.simplex, x, t, False, in_channels=img_channels)

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
        self.betas = betas
        self.sqrt_alphas = np.sqrt(alphas)
        self.sqrt_betas = np.sqrt(betas)

        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        # self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:],0.0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = ( betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev)* np.sqrt(alphas) / (1.0 - self.alphas_cumprod))
    def sample_t_with_weights(self, b_size, device):
        p = self.weights / np.sum(self.weights)
        indices_np = np.random.choice(len(p), size=b_size, p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / len(p) * p[indices_np]
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights



    def predict_x_0_from_eps(self, x_t, t, eps):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape, x_t.device) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, x_t.device) * eps)

    def predict_eps_from_x_0(self, x_t, t, pred_x_0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape, x_t.device) * x_t
                - pred_x_0) \
            / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, x_t.device)

    def q_mean_variance(self, x_0, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
                extract(self.sqrt_alphas_cumprod, t, x_0.shape, x_0.device) * x_0
        )
        variance = extract(1.0 - self.alphas_cumprod, t, x_0.shape, x_0.device)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_0.shape, x_0.device
        )
        return mean, variance, log_variance

    def p_mean_variance(self, model, x_t, t, estimate_noise=None):
        """
        Finds the mean & variance from N(x_{t-1}; mu_theta(x_t,t), sigma_theta (x_t,t))
        :param model:
        :param x_t:
        :param t:
        :return:
        """
        if estimate_noise == None:
            estimate_noise = model(x_t, t)

        # fixed model variance defined as \hat{\beta}_t - could add learned parameter
        model_var = np.append(self.posterior_variance[1], self.betas[1:])
        model_logvar = np.log(model_var)
        model_var = extract(model_var, t, x_t.shape, x_t.device)
        model_logvar = extract(model_logvar, t, x_t.shape, x_t.device)

        pred_x_0 = self.predict_x_0_from_eps(x_t, t, estimate_noise).clamp(-1, 1)


        model_mean, _, model_logvar_ = self.q_posterior_mean_variance(pred_x_0, x_t, t)
        return {"mean": model_mean,
               "variance": model_var,
               "log_variance": model_logvar,
            "pred_x_0": pred_x_0,}

    def sample_p(self, model, x_t, t, denoise_fn="gauss"):
        out = self.p_mean_variance(model, x_t, t)
        # noise = torch.randn_like(x_t)
        if type(denoise_fn) == str:
            if denoise_fn == "gauss":
                noise = torch.randn_like(x_t)
            elif denoise_fn == "noise_fn":
                noise = self.noise_fn(x_t, t).float()
            elif denoise_fn == "random":
                # noise = random_noise(self.simplex, x_t, t).float()
                noise = torch.randn_like(x_t)
            else:
                noise = generate_simplex_noise(self.simplex, x_t, t, False, in_channels=self.img_channels).float()
        else:
            noise = denoise_fn(x_t, t)

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        )
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_x_0": out["pred_x_0"]}

    def kl_loss(self, model, x_0, x_t, t, model_kwargs=None):

        # 1) posterior (scheduling)
        posterior_mean, _, posterior_log_var_clipped = self.q_posterior_mean_variance(x_0, x_t, t)

        # 2) prior (model)
        model_predict = self.p_mean_variance(model, x_t, t, estimate_noise=None)
        model_x_t_1_mean, model_x_t_1_log_var = model_predict['mean'], model_predict['log_variance']

        kl = normal_kl(posterior_mean, posterior_log_var_clipped,
                       model_x_t_1_mean, model_x_t_1_log_var)
        kl = mean_flat(kl) / np.log(2.0)
        return kl

    def forward_backward(self, model, x, see_whole_sequence="half", t_distance=None, denoise_fn="gauss",):
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
            x = self.sample_q(
                x, t_tensor,
                self.noise_fn(x, t_tensor).float()
            )
            if see_whole_sequence == "half":
                seq.append(x.cpu().detach())

        for t in range(int(t_distance) - 1, -1, -1):
            t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
            with torch.no_grad():
                out = self.sample_p(model, x, t_batch, denoise_fn)
                x = out["sample"]
            if see_whole_sequence:
                seq.append(x.cpu().detach())

        return x.detach() if not see_whole_sequence else seq
    def sample_q(self, x_0, t, noise):
        """
            q (x_t | x_0 )

            :param x_0:
            :param t:
            :param noise:
            :return:
        """
        return (extract(self.sqrt_alphas_cumprod, t, x_0.shape, x_0.device) * x_0 +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape, x_0.device) * noise)

    def step(self, model, x_t, t, noise):
        out = self.p_mean_variance(model, x_t, t)
        pred_x_0 = out['pred_x_0']


        posterior_mean = (extract(self.posterior_mean_coef1, t, x_t.shape, x_t.device) * pred_x_0
                        + extract(self.posterior_mean_coef2, t, x_t.shape, x_t.device) * x_t)
        posterior_log_var_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape, x_t.device)
        #sample = posterior_mean + torch.exp(0.5 * posterior_log_var_clipped) * noise
        sample = posterior_mean + torch.exp(0.5 * posterior_log_var_clipped) * torch.randn_like(x_t)
        return sample

    def step2(self, model, x_t, t, noise):
        out = self.p_mean_variance(model, x_t, t)
        pred_x_0 = out['pred_x_0']
        a = np.sqrt(extract(self.alphas_cumprod_prev, t, x_t.shape, x_t.device)) * pred_x_0
        b = extract(self.sqrt_one_minus_alphas_cumprod, t , x_t.shape, x_t.device) * noise
        posterior_log_var_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape, x_t.device)
        c = torch.exp(0.5 * posterior_log_var_clipped) * torch.randn_like(x_t)

        sample = a + b + c
        return sample

    """
    def step(self,
             model_output: torch.FloatTensor,
             timestep: int,
             sample: torch.FloatTensor, # x_t
             generator=None,
             return_dict: bool = True,
             ) -> Union[DDPMSchedulerOutput, Tuple]:
        t = timestep
        prev_t = self.previous_timestep(t)
        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None
        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        # 3. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff =          current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        # 6. Add noise
        variance = 0
        if t > 0:
            device = model_output.device
            variance_noise = randn_tensor(
                model_output.shape, generator=generator, device=device, dtype=model_output.dtype
            )
            if self.variance_type == "fixed_small_log":
                variance = self._get_variance(t, predicted_variance=predicted_variance) * variance_noise
            elif self.variance_type == "learned_range":
                variance = self._get_variance(t, predicted_variance=predicted_variance)
                variance = torch.exp(0.5 * variance) * variance_noise
            else:
                variance = (self._get_variance(t, predicted_variance=predicted_variance) ** 0.5) * variance_noise

        pred_prev_sample = pred_prev_sample + variance

        if not return_dict:
            return (pred_prev_sample,)

        return DDPMSchedulerOutput(prev_sample=pred_prev_sample,
                                   pred_original_sample=pred_original_sample)
    """
    """
    def ddim_sample(self,model,x,t,eta=0.0,):
        
        #Sample x_{t-1} from the model using DDIM.
        #Same usage as p_sample().
        
        out = self.p_mean_variance(model, x_t, t)
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self.predict_eps_from_x_0(x, t, out["pred_x_0"])

        alpha_bar = extract(self.alphas_cumprod, t, x_t.shape, x_t.device)
        alpha_bar_prev = extract(self.alphas_cumprod_prev, t, x_t.shape, x_t.device)

        sigma = (eta * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * th.sqrt(1 - alpha_bar / alpha_bar_prev))
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (out["pred_xstart"] * th.sqrt(alpha_bar_prev)) + (th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1))))  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}
    """
    """
    def step2(self, model, x_t, t):

        estimate_noise = model(x_t, t)
        #model_var = np.append(self.posterior_variance[1], self.betas[1:])
        posterior_mean_coef1 = extract(self.posterior_mean_coef1, t, x_t.shape, x_t.device)
        sqrt_recip_alphas_cumprod = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape, x_t.device)
        sqrt_recipm1_alphas_cumprod = extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, x_t.device)
        posterior_log_var_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape, x_t.device)
        posterior_mean_coef2 = extract(self.posterior_mean_coef2, t, x_t.shape, x_t.device)
        #sample = posterior_mean_coef1 * ((sqrt_recip_alphas_cumprod * x_t - sqrt_recipm1_alphas_cumprod * estimate_noise).clamp(-1, 1)) + posterior_mean_coef2 * x_t + \
        #         torch.exp(0.5 * posterior_log_var_clipped) * estimate_noise
        a = (posterior_mean_coef1 * sqrt_recip_alphas_cumprod + posterior_mean_coef2 ) * x_t
        b = (- posterior_mean_coef1 * sqrt_recipm1_alphas_cumprod + torch.exp(0.5 * posterior_log_var_clipped)) * estimate_noise
        #sample = (posterior_mean_coef1 *sqrt_recip_alphas_cumprod * x_t - posterior_mean_coef1 *sqrt_recipm1_alphas_cumprod * estimate_noise + posterior_mean_coef2 * x_t +  torch.exp(0.5 * posterior_log_var_clipped) * estimate_noise
        sample = a + b
        return sample
    """



    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """

        # mu (x_t,x_0) = \frac{\sqrt{alphacumprod prev} betas}{1-alphacumprod} *x_0
        # + \frac{\sqrt{alphas}(1-alphacumprod prev)}{ 1- alphacumprod} * x_t
        posterior_mean = (extract(self.posterior_mean_coef1, t, x_t.shape, x_t.device) * x_0
                          + extract(self.posterior_mean_coef2, t, x_t.shape, x_t.device) * x_t)
        # var = \frac{1-alphacumprod prev}{1-alphacumprod} * betas
        posterior_var = extract(self.posterior_variance, t, x_t.shape, x_t.device)
        posterior_log_var_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape, x_t.device)
        return posterior_mean, posterior_var, posterior_log_var_clipped
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
    def calc_vlb_xt(self, model, x_0, x_t, t, estimate_noise=None):
        # find KL divergence at t
        true_mean, _, true_log_var = self.q_posterior_mean_variance(x_0, x_t, t)
        output = self.p_mean_variance(model, x_t, t, estimate_noise)
        kl = normal_kl(true_mean, true_log_var, output["mean"], output["log_variance"])
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretised_gaussian_log_likelihood(
            x_0, output["mean"], log_scales=0.5 * output["log_variance"]
        )
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        nll = torch.where((t == 0), decoder_nll, kl)
        return {"output": nll, "pred_x_0": output["pred_x_0"]}
    def calc_loss(self, model, x_0, t):
        # noise = torch.randn_like(x)

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

    def p_loss(self, model, x_0, args):
        if self.loss_weight == "none":
            if args["train_start"]:
                t = torch.randint(
                    0, min(args["sample_distance"], self.num_timesteps), (x_0.shape[0],),
                    device=x_0.device
                )
            else:
                t = torch.randint(0, self.num_timesteps, (x_0.shape[0],), device=x_0.device)
            weights = 1
        else:
            t, weights = self.sample_t_with_weights(x_0.shape[0], x_0.device)

        loss, x_t, eps_t = self.calc_loss(model, x_0, t)
        loss = ((loss["loss"] * weights).mean(), (loss, x_t, eps_t))
        return loss
    def prior_vlb(self, x_0, args):
        t = torch.tensor([self.num_timesteps - 1] * args["Batch_Size"], device=x_0.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_0, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=torch.tensor(0.0, device=x_0.device),
            logvar2=torch.tensor(0.0, device=x_0.device)
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_total_vlb(self, x_0, model, args):
        vb = []
        x_0_mse = []
        mse = []
        for t in reversed(list(range(self.num_timesteps))):
            t_batch = torch.tensor([t] * args["Batch_Size"], device=x_0.device)
            noise = torch.randn_like(x_0)
            x_t = self.sample_q(x_0=x_0, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with torch.no_grad():
                out = self.calc_vlb_xt(
                    model,
                    x_0=x_0,
                    x_t=x_t,
                    t=t_batch,
                )
            vb.append(out["output"])
            x_0_mse.append(mean_flat((out["pred_x_0"] - x_0) ** 2))
            eps = self.predict_eps_from_x_0(x_t, t_batch, out["pred_x_0"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = torch.stack(vb, dim=1)
        x_0_mse = torch.stack(x_0_mse, dim=1)
        mse = torch.stack(mse, dim=1)

        prior_vlb = self.prior_vlb(x_0, args)
        total_vlb = vb.sum(dim=1) + prior_vlb
        return {
            "total_vlb": total_vlb,
            "prior_vlb": prior_vlb,
            "vb": vb,
            "x_0_mse": x_0_mse,
            "mse": mse,
        }

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

            for t_distance in range(50, int(args["T"] * 0.6), 50):
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
