python unet_ldm_pipeline_training.py --wandb_project_name 'text' --wandb_run_name "test" \
    --device "cuda:1" --experiment_dir '/data7/sooyeon/medical_image/anoddpm_result_ldm/text' \
    --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
    --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
    --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
    --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
    --img_size '128,128' --batch_size 6 --pretrained_vae_dir '/data7/sooyeon/medical_image/anoddpm_result_vae/2_ldm_pipeline_vae/vae/vae_76.pth' \
    --latent_channels 4 --sample_distance 150