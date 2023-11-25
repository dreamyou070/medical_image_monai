[screen 00]
python unet_training.py --wandb_project_name 'anoddpm_result_ldm' --wandb_run_name '1_masked_loss_latent' --device 'cuda:0' \
                        --experiment_dir '/data7/sooyeon/medical_image/anoddpm_result_ldm/1_masked_loss_latent' \
                        --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
                        --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
                        --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
                        --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
                        --model_save_freq 50 --img_size '128,128' --batch_size 6 --masked_loss_latent

[screen 01]
python unet_training.py --wandb_project_name 'anoddpm_result_ldm' --wandb_run_name '2_anormal_scoring' --device 'cuda:1' \
                        --experiment_dir '/data7/sooyeon/medical_image/anoddpm_result_ldm/2_anormal_scoring' \
                        --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
                        --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
                        --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
                        --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
                        --model_save_freq 50 --img_size '128,128' --batch_size 6 --anormal_scoring

[screen 02]
python unet_training.py --wandb_project_name 'anoddpm_result_ldm' --wandb_run_name '3_min_max_training' --device 'cuda:2' \
                        --experiment_dir '/data7/sooyeon/medical_image/anoddpm_result_ldm/3_min_max_training' \
                        --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
                        --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
                        --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
                        --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
                        --model_save_freq 50 --img_size '128,128' --batch_size 6 --min_max_training











[screen 02]
python unet_training.py --wandb_project_name 'anoddpm_result_ldm' --wandb_run_name '2_anormal_scoring' --device 'cuda:2' \
                        --experiment_dir '/data7/sooyeon/medical_image/anoddpm_result_ldm/2_anormal_scoring' \
                        --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
                        --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
                        --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
                        --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
                        --model_save_freq 50 --img_size '128,128' --batch_size 6 --anormal_scoring

[screen 04]
python unet_training.py --wandb_project_name 'anoddpm_result_ldm' --wandb_run_name '3_masked_loss_latent' --device 'cuda:3' \
                        --experiment_dir '/data7/sooyeon/medical_image/anoddpm_result_ldm/3_masked_loss_latent' \
                        --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
                        --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
                        --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
                        --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
                        --model_save_freq 50 --img_size '128,128' --batch_size 6 --masked_loss_latent








