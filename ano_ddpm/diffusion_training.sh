python diffusion_training.py --device cuda:0 \
         --wandb_api_key '3a3bc2f629692fa154b9274a5bbe5881d47245dc' \
         --wandb_project_name 'dental_experiment_anoddpm' --wandb_run_name '1_only_normal_training_gaussian' \
         --experiment_dir /data7/sooyeon/medical_image/anoddpm_result/1_only_normal_training_gaussian \
         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
         --img_siz '128,128' --batch_size 6 --train_start --save_imgs --sample_distance 150 --loss_type 'l2' --only_normal_training \
         --inference_num 4 --inference_freq 10 --vlb_freq 1 --model_save_freq 50 --save_imgs --train_epochs 3000
# ------------------------------------------------------------------------------------------------------------------------------------------------------
python diffusion_training.py --device cuda:1 \
         --wandb_api_key '3a3bc2f629692fa154b9274a5bbe5881d47245dc' \
         --wandb_project_name 'dental_experiment_anoddpm' --wandb_run_name '2_compare_allsample_nonmasked_loss' \
         --experiment_dir /data7/sooyeon/medical_image/anoddpm_result/2_compare_allsample_nonmasked_loss \
         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
         --img_siz '128,128' --batch_size 6 --train_start --save_imgs --sample_distance 150 --loss_type 'l2' \
         --inference_num 4 --inference_freq 10 --vlb_freq 1 --model_save_freq 50 --save_imgs --train_epochs 3000
# ------------------------------------------------------------------------------------------------------------------------------------------------------
python diffusion_training.py --device cuda:2 \
         --wandb_api_key '3a3bc2f629692fa154b9274a5bbe5881d47245dc' \
         --wandb_project_name 'dental_experiment_anoddpm' --wandb_run_name '3_masked_loss_gaussian' \
         --experiment_dir /data7/sooyeon/medical_image/anoddpm_result/3_masked_loss_gaussian \
         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
         --img_siz '128,128' --batch_size 6 --train_start --save_imgs --sample_distance 150 --loss_type 'l2' --masked_loss \
         --inference_num 4 --inference_freq 10 --vlb_freq 1 --model_save_freq 50 --save_imgs --train_epochs 3000


         --beta_schedule
# ------------------------------------------------------------------------------------------------------------------------------------------------------
[Screen 05]
python diffusion_training.py --device cuda:3 \
         --wandb_api_key '3a3bc2f629692fa154b9274a5bbe5881d47245dc' \
         --wandb_project_name 'dental_experiment_anoddpm' --wandb_run_name '4_pos_neg_loss_scale_1' \
         --experiment_dir /data7/sooyeon/medical_image/anoddpm_result/4_pos_neg_loss_scale_1 \
         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
         --img_siz '128,128' --batch_size 6 --train_start --save_imgs --sample_distance 150 --loss_type 'l2' --pos_neg_loss --pos_neg_loss_scale 1.0 \
         --inference_num 4 --inference_freq 10 --vlb_freq 1 --model_save_freq 50 --save_imgs --train_epochs 3000
# ------------------------------------------------------------------------------------------------------------------------------------------------------
[Screen 06]
python diffusion_training.py --device cuda:4 \
         --wandb_api_key '3a3bc2f629692fa154b9274a5bbe5881d47245dc' \
         --wandb_project_name 'dental_experiment_anoddpm' --wandb_run_name '5_pos_neg_loss_scale_2' \
         --experiment_dir /data7/sooyeon/medical_image/anoddpm_result/5_pos_neg_loss_scale_2 \
         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original' \
         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask' \
         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask' \
         --img_siz '128,128' --batch_size 6 --train_start --save_imgs --sample_distance 150 --loss_type 'l2' --pos_neg_loss --pos_neg_loss_scale 2.0 \
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



