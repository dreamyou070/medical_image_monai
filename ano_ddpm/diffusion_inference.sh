python diffusion_inference.py --device 'cuda:1' --process_title 'parksooyeon' \
                              --wandb_project_name 'inference_test' --wandb_run_name 'dental_inference_step_check' \
                              --experiment_dir '/data7/sooyeon/medical_image/anoddpm_result_ddpm/4_2_gaussian_cosine_infonce' \
                              --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_256/train/original' \
                              --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_256/train/mask' \
                              --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_256/valid/original' \
                              --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_256/valid/mask' \
                              --unet_pretrained_dir '/data7/sooyeon/medical_image/anoddpm_result_ddpm/4_2_gaussian_cosine_infonce/diffusion-models/unet_epoch_850.pt' \
                              --img_size '128,128' --batch_size 1 --beta_schedule 'linear'