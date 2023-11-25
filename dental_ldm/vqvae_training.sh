python vqvae_training.py --experiment_dir '/data7/sooyeon/medical_image/anoddpm_result_vqvae/1_first_trial' \
                         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
                         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
                         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
                         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
                         --device 'cuda:0' --img_size '128,128' --batch_size 6 --model_save_base_epoch 50
                         --