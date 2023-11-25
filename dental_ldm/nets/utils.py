
def update_ema_params(target, # ema
                      source, # unet
                      decay_rate=0.9999):
    targParams = dict(target.named_parameters())
    srcParams = dict(source.named_parameters())
    for k in targParams:
        # ------------------------------------------------------------------------------------------
        # new weight
        src_data = srcParams[k].data
        # ------------------------------------------------------------------------------------------
        # ema add only small number of present value
        targParams[k].data.mul_(decay_rate).add_(src_data,
                                                 alpha=1 - decay_rate)


