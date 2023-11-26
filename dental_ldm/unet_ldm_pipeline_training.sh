[Screen 04]
python unet_ldm_pipeline_training.py --wandb_project_name 'anoddpm_result_ldm_diffusers' --wandb_run_name "4_infonce_loss_sample_distance_250" \
    --device "cuda:4" --experiment_dir '/data7/sooyeon/medical_image/anoddpm_result_ldm_diffusers/4_infonce_loss_sample_distance_150' \
    --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/train/original' \
    --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/train/mask' \
    --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/valid/original' \
    --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/valid/mask' \
    --img_size '256,256' --batch_size 6 --pretrained_vae_dir '/data7/sooyeon/medical_image/pretrained/vae/vae_85.pth' \
    --sample_distance 250 --n_epochs 3000 --inference_freq 30 --model_save_freq 50 --infonce_loss

[Screen 05]
python unet_ldm_pipeline_training.py --wandb_project_name 'anoddpm_result_ldm_diffusers' --wandb_run_name "5_infonce_loss_sample_distance_250" \
    --device "cuda:5" --experiment_dir '/data7/sooyeon/medical_image/anoddpm_result_ldm_diffusers/5_infonce_loss_sample_distance_250' \
    --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/train/original' \
    --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/train/mask' \
    --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/valid/original' \
    --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/valid/mask' \
    --img_size '256,256' --batch_size 6 --pretrained_vae_dir '/data7/sooyeon/medical_image/anoddpm_result_vae/3_ldm_pipeline_vae_pixel_256_latent_32/vae/vae_51.pth' \
    --latent_channels 4 --sample_distance 250 --n_epochs 3000 --inference_freq 30 --model_save_freq 50 --infonce_loss

[Screen 04]
python mask_concating.py --wandb_project_name 'test' --wandb_run_name "4_infonce_loss_sample_distance_250" \
    --device "cuda:4" --experiment_dir '/data7/sooyeon/medical_image/anoddpm_result_ldm_diffusers/4_infonce_loss_sample_distance_150' \
    --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/train/original' \
    --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/train/mask' \
    --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/valid/original' \
    --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/valid/mask' \
    --img_size '256,256' --batch_size 6 --pretrained_vae_dir '/data7/sooyeon/medical_image/pretrained/vae/vae_85.pth' \
    --sample_distance 250 --n_epochs 3000 --inference_freq 30 --model_save_freq 50 --infonce_loss
