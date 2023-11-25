python vqvae_diffusion.py --wandb_project_name 'vqvae_diffusion' --wandb_run_name 'test' --device 'cuda:0' \
                        --experiment_dir '/data7/sooyeon/medical_image/vqvae_diffusion/1_test' \
                        --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
                        --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
                        --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
                        --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
                        --pretrained_vqvae_dir '/data7/sooyeon/medical_image/anoddpm_result_vqvae/1_first_trial/vqvae/vqvae_99.pth' \
                        --model_save_freq 50 --img_size '128,128' --batch_size 6 --latent_channels 32 \
                        --masked_loss_latent