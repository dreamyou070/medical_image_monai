python generate_heatmap.py --device 'cuda:6' \
                           --experiment_dir '/data7/sooyeon/medical_image/anoddpm_result/4_pos_infonce_loss' \
                           --model_name 'unet_global_step_24254.pt' \
                           --img_size '128,128' \
                           --batch_size 10 \
                           --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
                           --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
                           --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
                           --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
                           --thredhold 0.00000216