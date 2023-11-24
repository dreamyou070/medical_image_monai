[Screen 01]
python diffusion_training_infonce.py --device cuda:0 \
         --wandb_api_key '3a3bc2f629692fa154b9274a5bbe5881d47245dc' \
         --wandb_project_name 'dental_experiment_anoddpm' --wandb_run_name '4_pos_infonce_loss' \
         --experiment_dir /data7/sooyeon/medical_image/anoddpm_result/4_pos_infonce_loss \
         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
         --img_siz '128,128' --batch_size 6 --train_start --save_imgs --sample_distance 150 --loss_type 'l2' --beta_schedule 'linear' \
         --pos_infonce_loss \
         --neg_loss_scale 1 --inference_num 4 --inference_freq 10 --vlb_freq 1 --save_base_epoch 180 --save_imgs --train_epochs 300
[Screen 02]
python diffusion_training_infonce.py --device cuda:1 \
         --wandb_api_key '3a3bc2f629692fa154b9274a5bbe5881d47245dc' \
         --wandb_project_name 'dental_experiment_anoddpm' --wandb_run_name '5_infonce_loss' \
         --experiment_dir /data7/sooyeon/medical_image/anoddpm_result/5_infonce_loss \
         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
         --img_siz '128,128' --batch_size 6 --train_start --save_imgs --sample_distance 150 --loss_type 'l2' --beta_schedule 'linear' \
         --infonce_loss --neg_loss_scale 1.0 --inference_num 4 --inference_freq 10 --vlb_freq 1 --save_base_epoch 180 --save_imgs --train_epochs 300