[Screen 00]
python unet_ldm_pipeline_training.py --wandb_project_name 'anoddpm_result_ldm_diffusers' --wandb_run_name "1_circle_mask_masked_loss" \
    --device "cuda:4" --experiment_dir '/data7/sooyeon/medical_image/anoddpm_result_ldm_diffusers/' \
    --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_256/train/original' \
    --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_256/train/mask_circle' \
    --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_256/valid/original' \
    --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_256/valid/mask_circle' \
    --img_size '256,256' --batch_size 6 \
    --vae_config_dir '/data7/sooyeon/medical_image/pretrained/vae/config_3.json' \
    --pretrained_vae_dir '/data7/sooyeon/medical_image/pretrained/vae/1_vae_pixel_256_latent_32_latent_3_posterior_true_vae_74.pth' \
    --unet_config_dir '/data7/sooyeon/medical_image/pretrained/unet/config_3.json' \
    --sample_posterior --masked_loss \
    --sample_distance 250 --n_epochs 3000 --inference_freq 30 --model_save_freq 50