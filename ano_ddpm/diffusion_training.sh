# ------------------------------------------------------------------------------------------------------------------------------------------------------
[Screen 02]
python diffusion_training.py --device cuda:0 \
         --wandb_api_key '3a3bc2f629692fa154b9274a5bbe5881d47245dc' \
         --wandb_project_name 'dental_experiment_anoddpm' --wandb_run_name '3_1_masked_loss_gaussian_linear_scheduling' \
         --experiment_dir /data7/sooyeon/medical_image/anoddpm_result/3_1_masked_loss_gaussian_linear_scheduling \
         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
         --img_siz '128,128' --batch_size 6 --train_start --save_imgs --sample_distance 150 --loss_type 'l2' --masked_loss --beta_schedule 'linear' \
         --inference_num 4 --inference_freq 10 --vlb_freq 1 --model_save_freq 50 --save_imgs --train_epochs 3000
# ------------------------------------------------------------------------------------------------------------------------------------------------------
[Screen 04]
python diffusion_training.py --device cuda:3 \
         --wandb_api_key '3a3bc2f629692fa154b9274a5bbe5881d47245dc' \
         --wandb_project_name 'dental_experiment_anoddpm' --wandb_run_name '3_2_masked_loss_gaussian_cosine_scheduling' \
         --experiment_dir /data7/sooyeon/medical_image/anoddpm_result/3_2_masked_loss_gaussian_cosine_scheduling \
         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
         --img_siz '128,128' --batch_size 6 --train_start --save_imgs --sample_distance 150 --loss_type 'l2' --masked_loss --beta_schedule 'cosine' \
         --inference_num 4 --inference_freq 10 --vlb_freq 1 --model_save_freq 50 --save_imgs --train_epochs 3000
# ------------------------------------------------------------------------------------------------------------------------------------------------------
[Screen 06]
python diffusion_training.py --device cuda:1 \
         --wandb_api_key '3a3bc2f629692fa154b9274a5bbe5881d47245dc' \
         --wandb_project_name 'dental_experiment_anoddpm' --wandb_run_name '3_3_masked_loss_gaussian_sinusoidal_scheduling' \
         --experiment_dir /data7/sooyeon/medical_image/anoddpm_result/3_3_masked_loss_gaussian_sinusoidal_scheduling \
         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
         --img_siz '128,128' --batch_size 6 --train_start --save_imgs --sample_distance 150 --loss_type 'l2' --masked_loss --beta_schedule 'sinusoidal' \
         --inference_num 4 --inference_freq 10 --vlb_freq 1 --model_save_freq 50 --save_imgs --train_epochs 3000
# ------------------------------------------------------------------------------------------------------------------------------------------------------
[Screen 08]
python diffusion_training.py --device cuda:2 \
         --wandb_api_key '3a3bc2f629692fa154b9274a5bbe5881d47245dc' \
         --wandb_project_name 'dental_experiment_anoddpm' --wandb_run_name '3_3_2_masked_loss_gaussian_sinusoidal_scheduling_distance_80' \
         --experiment_dir /data7/sooyeon/medical_image/anoddpm_result/3_3_2_masked_loss_gaussian_sinusoidal_scheduling_distance_80 \
         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
         --img_siz '128,128' --batch_size 6 --train_start --save_imgs --sample_distance 80 --loss_type 'l2' --masked_loss --beta_schedule 'sinusoidal' \
         --inference_num 4 --inference_freq 10 --vlb_freq 1 --model_save_freq 50 --save_imgs --train_epochs 3000
# ------------------------------------------------------------------------------------------------------------------------------------------------------
[Screen 09]
python diffusion_training_infonce.py --device cuda:4 \
         --wandb_api_key '3a3bc2f629692fa154b9274a5bbe5881d47245dc' \
         --wandb_project_name 'dental_experiment_anoddpm' --wandb_run_name '4_gaussian_linear_infonce' \
         --experiment_dir /data7/sooyeon/medical_image/anoddpm_result/4_gaussian_linear_infonce \
         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
         --img_siz '128,128' --batch_size 6 --train_start --save_imgs --sample_distance 150 --loss_type 'l2' --beta_schedule 'linear' \
         --inference_num 4 --inference_freq 10 --vlb_freq 1 --model_save_freq 50 --save_imgs --train_epochs 3000 --infonce_loss
# ------------------------------------------------------------------------------------------------------------------------------------------------------
[Screen 10]
python diffusion_training_infonce.py --device cuda:5 \
         --wandb_api_key '3a3bc2f629692fa154b9274a5bbe5881d47245dc' \
         --wandb_project_name 'dental_experiment_anoddpm' --wandb_run_name '4_2_gaussian_cosine_infonce' \
         --experiment_dir /data7/sooyeon/medical_image/anoddpm_result/4_2_gaussian_cosine_infonce \
         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
         --img_siz '128,128' --batch_size 6 --train_start --save_imgs --sample_distance 150 --loss_type 'l2' --beta_schedule 'cosine' \
         --inference_num 4 --inference_freq 10 --vlb_freq 1 --model_save_freq 50 --save_imgs --train_epochs 3000 --infonce_loss
# ------------------------------------------------------------------------------------------------------------------------------------------------------
[Screen 11]
python diffusion_training_infonce.py --device cuda:6 \
         --wandb_api_key '3a3bc2f629692fa154b9274a5bbe5881d47245dc' \
         --wandb_project_name 'dental_experiment_anoddpm' --wandb_run_name '4_3_gaussian_sinusoidal_infonce' \
         --experiment_dir /data7/sooyeon/medical_image/anoddpm_result/4_3_gaussian_sinusoidal_infonce \
         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
         --img_siz '128,128' --batch_size 6 --train_start --save_imgs --sample_distance 150 --loss_type 'l2' --beta_schedule 'sinusoidal' \
         --inference_num 4 --inference_freq 10 --vlb_freq 1 --model_save_freq 50 --save_imgs --train_epochs 3000 --infonce_loss
# ------------------------------------------------------------------------------------------------------------------------------------------------------
[Screen 02]
python diffusion_training_infonce.py --device cuda:0 \
         --wandb_api_key '3a3bc2f629692fa154b9274a5bbe5881d47245dc' \
         --wandb_project_name 'dental_experiment_anoddpm' --wandb_run_name '4_1_2_gaussian_linear_infonce_scale_0.5' \
         --experiment_dir /data7/sooyeon/medical_image/anoddpm_result/4_1_2_gaussian_linear_infonce_scale_0.5 \
         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
         --img_siz '128,128' --batch_size 6 --train_start --save_imgs --sample_distance 150 --loss_type 'l2' --beta_schedule 'linear' \
         --neg_loss_scale 0.5 \
         --inference_num 4 --inference_freq 10 --vlb_freq 1 --model_save_freq 50 --save_imgs --train_epochs 3000 --infonce_loss
# ------------------------------------------------------------------------------------------------------------------------------------------------------
[Screen 04]
python diffusion_training_infonce.py --device cuda:1 \
         --wandb_api_key '3a3bc2f629692fa154b9274a5bbe5881d47245dc' \
         --wandb_project_name 'dental_experiment_anoddpm' --wandb_run_name '4_1_3_gaussian_linear_infonce_scale_2' \
         --experiment_dir /data7/sooyeon/medical_image/anoddpm_result/4_1_3_gaussian_linear_infonce_scale_2 \
         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
         --img_siz '128,128' --batch_size 6 --train_start --save_imgs --sample_distance 150 --loss_type 'l2' --beta_schedule 'linear' \
         --neg_loss_scale 2 \
         --inference_num 4 --inference_freq 10 --vlb_freq 1 --model_save_freq 50 --save_imgs --train_epochs 3000 --infonce_loss
# ------------------------------------------------------------------------------------------------------------------------------------------------------
[Screen 08]
python diffusion_training_infonce.py --device cuda:2 \
         --wandb_api_key '3a3bc2f629692fa154b9274a5bbe5881d47245dc' \
         --wandb_project_name 'dental_experiment_anoddpm' --wandb_run_name '7_gaussian_linear_pos_infonce' \
         --experiment_dir /data7/sooyeon/medical_image/anoddpm_result/7_gaussian_linear_pos_infonce \
         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
         --img_siz '128,128' --batch_size 6 --train_start --save_imgs --sample_distance 150 --loss_type 'l2' --beta_schedule 'linear' \
         --inference_num 4 --inference_freq 10 --vlb_freq 1 --model_save_freq 50 --save_imgs --train_epochs 3000 --pos_infonce_loss