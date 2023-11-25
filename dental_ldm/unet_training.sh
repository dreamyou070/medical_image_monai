[screen 00]
python unet_training.py --wandb_project_name 'anoddpm_result_ldm' --wandb_run_name '1_masked_loss' --device 'cuda:0' \
                        --experiment_dir '/data7/sooyeon/medical_image/anoddpm_result_ldm/1_masked_loss' \
                        --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
                        --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
                        --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
                        --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
                        --model_save_freq 50 --img_size '128,128' --batch_size 6 --masked_loss
                        
[screen 01]
python unet_training.py --wandb_project_name 'anoddpm_result_ldm' --wandb_run_name '2_info_nce_loss' --device 'cuda:1' \
                        --experiment_dir '/data7/sooyeon/medical_image/anoddpm_result_ldm/2_info_nce_loss' \
                        --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
                        --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
                        --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
                        --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
                        --model_save_freq 50 --img_size '128,128' --batch_size 6 --info_nce_loss

[screen 02]
python unet_training.py --wandb_project_name 'anoddpm_result_ldm' --wandb_run_name '3_pos_info_nce_loss' --device 'cuda:2' \
                        --experiment_dir '/data7/sooyeon/medical_image/anoddpm_result_ldm/3_pos_info_nce_loss' \
                        --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
                        --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
                        --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
                        --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
                        --model_save_freq 50 --img_size '128,128' --batch_size 6 --pos_info_nce_loss