[screen 00]
python vae_ldm_pipeline_training.py --wandb_project_name 'dental_vae_training' --wandb_run_name '0_vae_pixel_256_latent_32_latent_3_posterior_False' \
                        --device 'cuda:0' \
                        --experiment_dir '/data7/sooyeon/medical_image/anoddpm_result_vae/0_vae_pixel_256_latent_32_latent_3_posterior_False' \
                        --vae_config_dir '/data7/sooyeon/medical_image/pretrained/vae/config_3.json' \
                        --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_256/train/original' \
                        --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_256/train/mask' \
                        --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_256/valid/original' \
                        --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_256/valid/mask' \
                        --model_save_freq 50 --img_size '256,256' --batch_size 3 --loss_type 'l2' --kl_weight 0.000001 --perceptual_weight 0.001 --adv_weight 0.01
[screen 01]
python vae_ldm_pipeline_training.py --wandb_project_name 'dental_vae_training' --wandb_run_name '1_vae_pixel_256_latent_32_latent_3_posterior_true' \
                        --device 'cuda:1' \
                        --experiment_dir '/data7/sooyeon/medical_image/anoddpm_result_vae/1_vae_pixel_256_latent_32_latent_3_posterior_true' \
                        --vae_config_dir '/data7/sooyeon/medical_image/pretrained/vae/config_3.json' \
                        --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_256/train/original' \
                        --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_256/train/mask' \
                        --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_256/valid/original' \
                        --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_256/valid/mask' \
                        --model_save_freq 50 --img_size '256,256' --batch_size 3 --loss_type 'l2' --kl_weight 0.000001 --perceptual_weight 0.001 --adv_weight 0.01 --sample_posterior
[screen 02]
python vae_ldm_pipeline_training.py --wandb_project_name 'dental_vae_training' --wandb_run_name '2_vae_pixel_256_latent_32_latent_4_posterior_False' \
                        --device 'cuda:2' \
                        --experiment_dir '/data7/sooyeon/medical_image/anoddpm_result_vae/2_vae_pixel_256_latent_32_latent_4_posterior_False' \
                        --vae_config_dir '/data7/sooyeon/medical_image/pretrained/vae/config.json' \
                        --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_256/train/original' \
                        --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_256/train/mask' \
                        --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_256/valid/original' \
                        --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_256/valid/mask' \
                        --model_save_freq 50 --img_size '256,256' --batch_size 3 --loss_type 'l2' --kl_weight 0.000001 --perceptual_weight 0.001 --adv_weight 0.01
[screen 03]
python vae_ldm_pipeline_training.py --wandb_project_name 'dental_vae_training' --wandb_run_name '3_vae_pixel_256_latent_32_latent_4_posterior_true' \
                        --device 'cuda:3' \
                        --experiment_dir '/data7/sooyeon/medical_image/anoddpm_result_vae/3_vae_pixel_256_latent_32_latent_4_posterior_true' \
                        --vae_config_dir '/data7/sooyeon/medical_image/pretrained/vae/config.json' \
                        --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_256/train/original' \
                        --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_256/train/mask' \
                        --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_256/valid/original' \
                        --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_256/valid/mask' \
                        --model_save_freq 50 --img_size '256,256' --batch_size 3 --loss_type 'l2' --kl_weight 0.000001 --perceptual_weight 0.001 --adv_weight 0.01 --sample_posterior