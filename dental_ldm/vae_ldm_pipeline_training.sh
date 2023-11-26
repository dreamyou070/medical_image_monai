python vae_ldm_pipeline_training.py --wandb_project_name 'dental_vae_training' --wandb_run_name '3_ldm_pipeline_vae_pixel_256_latent_32' --device 'cuda:0' \
                        --experiment_dir '/data7/sooyeon/medical_image/anoddpm_result_vae/3_ldm_pipeline_vae_pixel_256_latent_32' \
                        --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/train/original' \
                        --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/train/mask' \
                        --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/valid/original' \
                        --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/valid/mask' \
                        --model_save_freq 50 --img_size '256,256' --batch_size 3 --masked_loss_latent --sample_distance 50







python vae_check.py --wandb_project_name 'test' --wandb_run_name 'test' --device 'cuda:1' \
                        --experiment_dir '/data7/sooyeon/medical_image/anoddpm_result_vae/test' \
                        --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/train/original' \
                        --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/train/mask' \
                        --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/valid/original' \
                        --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/valid/mask' \
                        --model_save_freq 50 --img_size '256,256' --batch_size 3 --masked_loss_latent --sample_distance 50