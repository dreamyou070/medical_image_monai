python image_inverting.py --wandb_project_name 'image_inverting_test' --wandb_run_name "image_inverting_test" \
    --seed 42 --device 'cuda:4' --img_size "256,256" --batch_size 1 \
    --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/train/original' \
    --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/train/mask' \
    --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/valid/original' \
    --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_512/valid/mask' \
    --pretrained_vae_dir '/data7/sooyeon/medical_image/pretrained/vae/vae_85.pth' \
    --pretrained_unet_dir '/data7/sooyeon/medical_image/anoddpm_result_ldm_diffusers/0_only_normal_training/diffusion-models/unet_epoch_200.pt' \
    --experiment_dir '/data7/sooyeon/medical_image/anoddpm_result_ldm_diffusers/0_only_normal_training'