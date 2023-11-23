[Screen 01]
python diffusion_training.py --device cuda:0 \
         --wandb_api_key '3a3bc2f629692fa154b9274a5bbe5881d47245dc' \
         --wandb_project_name 'dental_experiment_anoddpm' --wandb_run_name '3_masked_loss_gaussian' \
         --experiment_dir /data7/sooyeon/medical_image/anoddpm_result/3_masked_loss_gaussian \
         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
         --img_siz '128,128' --batch_size 6 --train_start --save_imgs --sample_distance 150 --loss_type 'l2' --masked_loss --beta_schedule 'linear' \
         --inference_num 4 --inference_freq 10 --vlb_freq 1 --model_save_freq 50 --save_imgs --train_epochs 3000
# ------------------------------------------------------------------------------------------------------------------------------------------------------
[Screen 07]
python diffusion_training.py --device cuda:6 \
         --wandb_api_key '3a3bc2f629692fa154b9274a5bbe5881d47245dc' \
         --wandb_project_name 'dental_experiment_anoddpm' --wandb_run_name '3_masked_loss_gaussian_change_whole_kl' \
         --experiment_dir /data7/sooyeon/medical_image/anoddpm_result/3_masked_loss_gaussian \
         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
         --img_siz '128,128' --batch_size 6 --train_start --save_imgs --sample_distance 150 --loss_type 'l2' --masked_loss --beta_schedule 'linear' \
         --inference_num 4 --inference_freq 10 --vlb_freq 1 --model_save_freq 50 --save_imgs --train_epochs 3000



[Screen 02]
python diffusion_training_pos_neg.py --device cuda:1 \
         --wandb_api_key '3a3bc2f629692fa154b9274a5bbe5881d47245dc' \
         --wandb_project_name 'dental_experiment_anoddpm' --wandb_run_name '4_pos_neg_loss_scale_1' \
         --experiment_dir /data7/sooyeon/medical_image/anoddpm_result/4_pos_neg_loss_scale_1 \
         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
         --img_siz '128,128' --batch_size 6 --train_start --save_imgs --sample_distance 150 --loss_type 'l2' --beta_schedule 'linear' --pos_neg_loss --pos_neg_loss_scale 1.0 \
         --inference_num 4 --inference_freq 10 --vlb_freq 1 --model_save_freq 50 --save_imgs --train_epochs 3000
# ------------------------------------------------------------------------------------------------------------------------------------------------------
[Screen 03]
python diffusion_training_pos_neg.py --device cuda:2 \
         --wandb_api_key '3a3bc2f629692fa154b9274a5bbe5881d47245dc' \
         --wandb_project_name 'dental_experiment_anoddpm' --wandb_run_name '5_pos_neg_loss_scale_2' \
         --experiment_dir /data7/sooyeon/medical_image/anoddpm_result/5_pos_neg_loss_scale_2 \
         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
         --img_siz '128,128' --batch_size 6 --train_start --save_imgs --sample_distance 150 --loss_type 'l2' --beta_schedule 'linear' --pos_neg_loss --pos_neg_loss_scale 2.0 \
         --inference_num 4 --inference_freq 10 --vlb_freq 1 --model_save_freq 50 --save_imgs --train_epochs 3000
# ------------------------------------------------------------------------------------------------------------------------------------------------------
[Screen 03]
python diffusion_training_pos_neg.py --device cuda:6 \
         --wandb_api_key '3a3bc2f629692fa154b9274a5bbe5881d47245dc' \
         --wandb_project_name 'dental_experiment_anoddpm' --wandb_run_name '5_2_pos_neg_loss_scale_2_cosine_schedule' \
         --experiment_dir /data7/sooyeon/medical_image/anoddpm_result/5_2_pos_neg_loss_scale_2_cosine_schedule \
         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
         --img_siz '128,128' --batch_size 6 --train_start --save_imgs --sample_distance 150 --loss_type 'l2' --beta_schedule 'cosine' --pos_neg_loss --pos_neg_loss_scale 2.0 \
         --inference_num 4 --inference_freq 10 --vlb_freq 1 --model_save_freq 50 --save_imgs --train_epochs 3000


[Screen 04]
python diffusion_training.py --device cuda:3 \
         --wandb_api_key '3a3bc2f629692fa154b9274a5bbe5881d47245dc' \
         --wandb_project_name 'dental_experiment_anoddpm' --wandb_run_name '3_2_masked_loss_gaussian_cosine_schedule' \
         --experiment_dir /data7/sooyeon/medical_image/anoddpm_result/3_2_masked_loss_gaussian_cosine_schedule \
         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
         --img_siz '128,128' --batch_size 6 --train_start --save_imgs --sample_distance 150 --loss_type 'l2' --masked_loss --beta_schedule 'cosine' \
         --inference_num 4 --inference_freq 10 --vlb_freq 1 --model_save_freq 50 --save_imgs --train_epochs 3000

[Screen 07]
python diffusion_training.py --device cuda:6 \
         --wandb_api_key '3a3bc2f629692fa154b9274a5bbe5881d47245dc' \
         --wandb_project_name 'dental_experiment_anoddpm' --wandb_run_name '3_2_2_use_simplex_noise' \
         --experiment_dir /data7/sooyeon/medical_image/anoddpm_result/3_2_2_masked_loss_gaussian_cosine_schedule_change_kl_cal \
         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
         --img_siz '128,128' --batch_size 6 --train_start --save_imgs --sample_distance 150 --loss_type 'l2' --masked_loss --beta_schedule 'cosine' \
         --use_simplex_noise \
         --inference_num 4 --inference_freq 10 --vlb_freq 1 --model_save_freq 50 --save_imgs --train_epochs 3000


[Screen 05]
python diffusion_training.py --device cuda:4 \
         --wandb_api_key '3a3bc2f629692fa154b9274a5bbe5881d47245dc' \
         --wandb_project_name 'dental_experiment_anoddpm' --wandb_run_name '3_3_masked_loss_gaussian_sinusoidal_schedule' \
         --experiment_dir /data7/sooyeon/medical_image/anoddpm_result/3_3_masked_loss_gaussian_sinusoidal_schedule \
         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
         --img_siz '128,128' --batch_size 6 --train_start --save_imgs --sample_distance 150 --loss_type 'l2' --masked_loss --beta_schedule 'sinusoidal' \
         --inference_num 4 --inference_freq 10 --vlb_freq 1 --model_save_freq 50 --save_imgs --train_epochs 3000

[Screen 06]
python diffusion_training.py --device cuda:5 \
         --wandb_api_key '3a3bc2f629692fa154b9274a5bbe5881d47245dc' \
         --wandb_project_name 'dental_experiment_anoddpm' --wandb_run_name '3_3_2_masked_loss_gaussian_sinusoidal_schedule_shortterm' \
         --experiment_dir /data7/sooyeon/medical_image/anoddpm_result/3_3_2_masked_loss_gaussian_sinusoidal_schedule_shortterm \
         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
         --img_siz '128,128' --batch_size 6 --train_start --save_imgs --sample_distance 50 --loss_type 'l2' --masked_loss --beta_schedule 'sinusoidal' \
         --inference_num 4 --inference_freq 10 --vlb_freq 1 --model_save_freq 50 --save_imgs --train_epochs 3000









# ------------------------------------------------------------------------------------------------------------------------------------------------------
[Screen 07]
python diffusion_training.py --device cuda:5 \
         --wandb_api_key '3a3bc2f629692fa154b9274a5bbe5881d47245dc' \
         --wandb_project_name 'dental_experiment_anoddpm' --wandb_run_name '6_pos_neg_loss_scale_4' \
         --experiment_dir /data7/sooyeon/medical_image/anoddpm_result/6_pos_neg_loss_scale_4 \
         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
         --img_siz '128,128' --batch_size 6 --train_start --save_imgs --sample_distance 150 --loss_type 'l2' --pos_neg_loss --pos_neg_loss_scale 4.0 \
         --inference_num 4 --inference_freq 10 --vlb_freq 1 --model_save_freq 50 --save_imgs --train_epochs 3000
# ------------------------------------------------------------------------------------------------------------------------------------------------------
[Sereen 01]
python diffusion_training.py --device cuda:0 \
         --wandb_api_key '3a3bc2f629692fa154b9274a5bbe5881d47245dc' \
         --wandb_project_name 'dental_experiment_anoddpm' --wandb_run_name '3_3_masked_loss_gaussian_extreme_cosine_schedule_short_timelength' \
         --experiment_dir /data7/sooyeon/medical_image/anoddpm_result/3_3_masked_loss_gaussian_extreme_cosine_schedule_short_timelength \
         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
         --img_siz '128,128' --batch_size 6 --train_start --save_imgs --sample_distance 550 --loss_type 'l2' --masked_loss --beta_schedule 'extreme_cosine' \
         --inference_num 4 --inference_freq 10 --vlb_freq 1 --model_save_freq 50 --save_imgs --train_epochs 3000
# ------------------------------------------------------------------------------------------------------------------------------------------------------

python diffusion_training.py --device cuda:6 \
         --wandb_api_key '3a3bc2f629692fa154b9274a5bbe5881d47245dc' \
         --wandb_project_name 'dental_experiment_anoddpm' --wandb_run_name '3_2_masked_loss_gaussian_extreme_cosine_schedule' \
         --experiment_dir /data7/sooyeon/medical_image/anoddpm_result/3_2_masked_loss_gaussian_extreme_cosine_schedule \
         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
         --img_siz '128,128' --batch_size 6 --train_start --save_imgs --sample_distance 150 --loss_type 'l2' --masked_loss --beta_schedule 'extreme_cosine' \
         --inference_num 4 --inference_freq 10 --vlb_freq 1 --model_save_freq 50 --save_imgs --train_epochs 3000