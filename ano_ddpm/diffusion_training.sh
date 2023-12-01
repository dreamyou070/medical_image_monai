[Screen 05]
python diffusion_training_vlb.py --device cuda:7 \
         --wandb_api_key '3a3bc2f629692fa154b9274a5bbe5881d47245dc' \
         --wandb_project_name 'MVTec_experiment' --wandb_run_name '1_first_experiment' \
         --experiment_dir /data7/sooyeon/medical_image/MVTec_result/1_first_experiment \
         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/MV/bagel/train/good/rgb' \
         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/MV/bagel/train/good/xyz' \
         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/MV/bagel/test/crack/rgb' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/MV/bagel/test/crack/xyz' \
         --img_siz '256,256' --batch_size 2 --train_start --save_imgs --sample_distance 300 --beta_schedule 'linear' \
         --in_channels 3 --inference_num 4 --train_epochs 300 --model_save_freq 5 --use_step1

[Screen 04]
python diffusion_training_vlb.py --device cuda:4 \
         --wandb_api_key '3a3bc2f629692fa154b9274a5bbe5881d47245dc' \
         --wandb_project_name 'anoddpm_result' --wandb_run_name '2_normal_training_kl_loss_step_infer' \
         --experiment_dir /data7/sooyeon/medical_image/anoddpm_result/2_normal_training_kl_loss_step_infer \
         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_256/train/original' \
         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_256/train/mask_rectangle_by_record' \
         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_256/valid/original' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_256/valid/mask_rectangle_by_record' \
         --img_siz '256,256' --batch_size 2 --train_start --save_imgs --sample_distance 300 --beta_schedule 'linear' \
         --inference_num 4 --train_epochs 300 --model_save_freq 30 --only_normal_training --use_vlb_loss --use_step1
