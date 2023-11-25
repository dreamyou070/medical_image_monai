python unet_training.py --wandb_project_name 'test' --wandb_run_name 'ldm_training' \
                        --device 'cuda1' \
                        --experiment_dir '/data7/sooyeon/medical_image/anoddpm_result/20231119_dental_test' \
                        --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
                        --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
                        --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
                        --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
                        --img_size '128,128' --batch_size 6