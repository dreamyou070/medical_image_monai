python vae_ldm_pipeline_training.py --wandb_project_name 'dental_vae_training' --wandb_run_name '3_ldm_pipeline_vae_laten_size_check' --device 'cuda:0' \
                        --experiment_dir '/data7/sooyeon/medical_image/anoddpm_result_vae/3_ldm_pipeline_vae_laten_size_check' \
                        --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
                        --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
                        --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
                        --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
                        --model_save_freq 50 --img_size '128,128' --batch_size 6 --masked_loss_latent --sample_distance 50