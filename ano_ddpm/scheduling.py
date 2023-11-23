from GaussianDiffusion_Sy import get_beta_schedule
from matplotlib import pyplot as plt
import numpy as np
import math


num_diffusion_steps = 1000
# ---------------------------------------------------------------
def get_beta_schedule(num_diffusion_steps, name="cosine"):
    betas = []

    if name == "cosine":
        max_beta = 0.999
        f = lambda t: np.cos((t + 0.008) / (1+0.008) * np.pi / 2) ** 2
        for i in range(num_diffusion_steps):
            t1 = i / num_diffusion_steps
            t2 = (i + 1) / num_diffusion_steps
            betas.append(min(1 - f(t2) / f(t1), max_beta))
        betas = np.array(betas)

    elif name == 'extreme_cosine' :
        max_beta = 0.999
        f = lambda t: (1 + np.cos((((t + 0.008) / (1 + 0.008) * np.pi / 2) + np.pi / 2))) ** 2
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
        max_beta = 0.999
        f = lambda t: (1 + np.sin((((t + 0.008) / (1 + 0.008) * np.pi / 2) + np.pi / 2))) ** 2
        for i in range(num_diffusion_steps):
            t1 = i / num_diffusion_steps
            t2 = (i + 1) / num_diffusion_steps
            betas.append(min(1 - f(t2) / f(t1), max_beta))
        betas = np.array(betas)

    elif name == 'new_sinusoidal' :
        # want green should be zero
        # that means alphas be zero
        max_beta = 0.999
        base_angle = np.pi / 2
        f = lambda t: np.sin(t * np.pi / 2)
        for i in range(num_diffusion_steps):
            t1 = i / num_diffusion_steps
            value = (f(t1) + 0.0008) / 2
            betas.append(value)
        betas = np.array(betas)

    #    raise NotImplementedError(f"unknown beta schedule: {name}")
    return betas

def get_alphas(betas) :
    alphas = 1 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    return alphas, alphas_cumprod
def timesteps(num_timesteps, timesteps_const=1000) :
    return np.linspace(0, num_timesteps, timesteps_const)


# variance_noise = 1- alphas_cumprod
# alpha_hat_T = alphas_cumprod[-1] # almost 0
# alpha_hat_0 = alphas_cumprod[0]  # almost 1
snr_compare = 'snr_compare'
def main() :

    print(f' (0) time steps')
    timestep = timesteps(1000)

    print(f' (1) linear scheduler (from small to big 0.02) ')
    betas = get_beta_schedule(num_diffusion_steps, name="linear")
    alphas, alphas_cumprod = get_alphas(betas)
    print(f'beta_0 : {betas[0]}  | alphas_cumprod_0 = {alphas_cumprod[0]}')
    print(f'beta_150 : {betas[150]}  | alphas_cumprod_150 = {alphas_cumprod[150]}')
    print(f'beta_T : {betas[-1]} | alphas_cumprod_T = {alphas_cumprod[-1]}')
    snr = alphas_cumprod / (1-alphas_cumprod)
    print(f'snr_0 : {snr[0]} | snr_150 : {snr[150]}')
    #plt.plot(timestep, snr, label='SNR')
    #plt.plot(timestep, betas, label='betas')
    plt.plot(timestep, alphas_cumprod, label='alphas_cumprod')
    plt.xlabel('timestep')
    plt.yscale('log', base=10)
    plt.title('linear schedule')
    plt.legend()
    plt.show()

    print(f'\n (2) cosine scheduler')
    betas = get_beta_schedule(num_diffusion_steps, name="cosine")
    alphas, alphas_cumprod = get_alphas(betas)
    snr = alphas_cumprod / (1 - alphas_cumprod)
    print(f'beta_0 : {betas[0]}  | alphas_cumprod_0 = {alphas_cumprod[0]}')
    print(f'beta_150 : {betas[150]}  | alphas_cumprod_150 = {alphas_cumprod[150]}')
    print(f'beta_T : {betas[-1]} | alphas_cumprod_T = {alphas_cumprod[-1]}')
    print(f'snr_0 : {snr[0]} | snr_150 : {snr[150]}')
    #plt.plot(timestep, snr, label='SNR')
    #plt.plot(timestep, betas, label='betas')
    #plt.plot(timestep, alphas_cumprod, label='alphas_cumprod')
    #plt.xlabel('timestep')
    #plt.yscale('log', base=10)
    #plt.title('cosine schedule SNR')
    #plt.legend()
    #plt.show()

    print(f'\n (3) extreme cosine')
    betas = get_beta_schedule(num_diffusion_steps, name="extreme_cosine")
    alphas, alphas_cumprod = get_alphas(betas)
    snr = alphas_cumprod / (1 - alphas_cumprod)
    #plt.plot(timestep, snr, label='SNR')
    #plt.plot(timestep, betas, label='betas')
    #plt.plot(timestep, alphas_cumprod, label='alphas_cumprod')
    #plt.xlabel('timestep')
    #plt.yscale('log', base=10)
    #plt.title('propose extreme cosine schedule')
    #plt.legend()
    #plt.show()

    print(f'\n (4) sinusoidal')
    betas = get_beta_schedule(num_diffusion_steps, name="sinusoidal")
    alphas, alphas_cumprod = get_alphas(betas)
    snr = alphas_cumprod / (1 - alphas_cumprod)
    print(f'beta_0 : {betas[0]}  | alphas_cumprod_0 = {alphas_cumprod[0]}')
    print(f'beta_150 : {betas[150]}  | alphas_cumprod_150 = {alphas_cumprod[150]}')
    print(f'beta_T : {betas[-1]} | alphas_cumprod_T = {alphas_cumprod[-1]}')
    print(f'snr_0 : {snr[0]} | snr_150 : {snr[150]}')
    # plt.plot(timestep, snr, label='SNR')
    # plt.plot(timestep, betas, label='betas')
    # plt.plot(timestep, alphas_cumprod, label='alphas_cumprod')
    # plt.xlabel('timestep')
    # plt.yscale('log', base=10)
    # plt.title('cosine schedule SNR')
    # plt.legend()
    # plt.show()

    print(f'\n (5) new sin')
    betas = get_beta_schedule(num_diffusion_steps, name="new_sinusoidal")
    alphas, alphas_cumprod = get_alphas(betas)
    print(f'beta_0 : {betas[0]}  | alphas_cumprod_0 = {alphas_cumprod[0]}')
    print(f'beta_150 : {betas[150]}  | alphas_cumprod_150 = {alphas_cumprod[150]}')
    print(f'beta_T : {betas[-1]} | alphas_cumprod_T = {alphas_cumprod[-1]}')
    print(f'snr_0 : {snr[0]} | snr_150 : {snr[150]}')
    snr = alphas_cumprod / (1 - alphas_cumprod)
    plt.plot(timestep, snr, label='SNR')
    plt.plot(timestep, betas, label='betas')
    plt.plot(timestep, alphas_cumprod, label='alphas_cumprod')
    plt.xlabel('timestep')
    plt.yscale('log', base=10)
    plt.title('sine schedule')
    plt.legend()
    plt.show()


if __name__ == '__main__' :
    main()
