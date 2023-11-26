[Screen 00]
python unet_ldm_pipeline_training.py --wandb_project_name 'anoddpm_result_ldm_diffusers' --wandb_run_name "0_only_normal_training" \
    --device "cuda:3" --experiment_dir '/data7/sooyeon/medical_image/anoddpm_result_ldm_diffusers/0_only_normal_training' \
    --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/train/original' \
    --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/train/mask' \
    --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/valid/original' \
    --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/valid/mask' \
    --img_size '256,256' --batch_size 6 --pretrained_vae_dir '/data7/sooyeon/medical_image/anoddpm_result_vae/3_ldm_pipeline_vae_pixel_256_latent_32/vae/vae_51.pth' \
    --latent_channels 4 --sample_distance 150 --n_epochs 3000 --inference_freq 50 --model_save_freq 50 --only_normal_training

[Screen 01]
python unet_ldm_pipeline_training.py --wandb_project_name 'anoddpm_result_ldm_diffusers' --wandb_run_name "1_only_normal_training_sample_distance_30" \
    --device "cuda:1" --experiment_dir '/data7/sooyeon/medical_image/anoddpm_result_ldm_diffusers/1_only_normal_training_sample_distance_30' \
    --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/train/original' \
    --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/train/mask' \
    --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/valid/original' \
    --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/valid/mask' \
    --img_size '256,256' --batch_size 6 --pretrained_vae_dir '/data7/sooyeon/medical_image/anoddpm_result_vae/3_ldm_pipeline_vae_pixel_256_latent_32/vae/vae_51.pth' \
    --latent_channels 4 --sample_distance 30 --n_epochs 3000 --inference_freq 50 --model_save_freq 50 --only_normal_training

[Screen 02]
python unet_ldm_pipeline_training.py --wandb_project_name 'anoddpm_result_ldm_diffusers' --wandb_run_name "2_masked_loss_sample_distance_150" \
    --device "cuda:2" --experiment_dir '/data7/sooyeon/medical_image/anoddpm_result_ldm_diffusers/2_masked_loss_sample_distance_150' \
    --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/train/original' \
    --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/train/mask' \
    --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/valid/original' \
    --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/valid/mask' \
    --img_size '256,256' --batch_size 6 --pretrained_vae_dir '/data7/sooyeon/medical_image/anoddpm_result_vae/3_ldm_pipeline_vae_pixel_256_latent_32/vae/vae_51.pth' \
    --latent_channels 4 --sample_distance 150 --n_epochs 3000 --inference_freq 50 --model_save_freq 50 --masked_loss

[Screen 03]
python unet_ldm_pipeline_training.py --wandb_project_name 'anoddpm_result_ldm_diffusers' --wandb_run_name "3_masked_loss_sample_distance_30" \
    --device "cuda:0" --experiment_dir '/data7/sooyeon/medical_image/anoddpm_result_ldm_diffusers/3_masked_loss_sample_distance_30' \
    --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/train/original' \
    --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/train/mask' \
    --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/valid/original' \
    --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/valid/mask' \
    --img_size '256,256' --batch_size 6 --pretrained_vae_dir '/data7/sooyeon/medical_image/anoddpm_result_vae/3_ldm_pipeline_vae_pixel_256_latent_32/vae/vae_51.pth' \
    --latent_channels 4 --sample_distance 30 --n_epochs 3000 --inference_freq 50 --model_save_freq 50 --masked_loss

[Screen 04]
python unet_ldm_pipeline_training.py --wandb_project_name 'anoddpm_result_ldm_diffusers' --wandb_run_name "4_infonce_loss_sample_distance_150" \
    --device "cuda:2" --experiment_dir '/data7/sooyeon/medical_image/anoddpm_result_ldm_diffusers/4_infonce_loss_sample_distance_150' \
    --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/train/original' \
    --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/train/mask' \
    --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/valid/original' \
    --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/valid/mask' \
    --img_size '256,256' --batch_size 6 --pretrained_vae_dir '/data7/sooyeon/medical_image/anoddpm_result_vae/3_ldm_pipeline_vae_pixel_256_latent_32/vae/vae_51.pth' \
    --latent_channels 4 --sample_distance 150 --n_epochs 3000 --inference_freq 50 --model_save_freq 50 --infonce_loss

[Screen 03]
python unet_ldm_pipeline_training.py --wandb_project_name 'anoddpm_result_ldm_diffusers' --wandb_run_name "3_masked_loss_sample_distance_30" \
    --device "cuda:0" --experiment_dir '/data7/sooyeon/medical_image/anoddpm_result_ldm_diffusers/3_masked_loss_sample_distance_30' \
    --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/train/original' \
    --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/train/mask' \
    --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/valid/original' \
    --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/valid/mask' \
    --img_size '256,256' --batch_size 6 --pretrained_vae_dir '/data7/sooyeon/medical_image/anoddpm_result_vae/3_ldm_pipeline_vae_pixel_256_latent_32/vae/vae_51.pth' \
    --latent_channels 4 --sample_distance 30 --n_epochs 3000 --inference_freq 50 --model_save_freq 50 --masked_loss