python diffusion_training.py --device cuda:5 \
         --wandb_api_key '3a3bc2f629692fa154b9274a5bbe5881d47245dc' \
         --wandb_project_name 'MVTec_experiment' --wandb_run_name '1_no_kl_loss_step_infer_diffuser_module' \
         --experiment_dir /data7/sooyeon/medical_image/MVTec_result/1_no_kl_loss_step_infer_diffuser_module \
         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/MV/bagel/train/good/rgb' \
         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/MV/bagel/train/good/rgb' \
         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/MV/bagel/test/crack/rgb' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/MV/bagel/test/crack/rgb' \
         --img_siz '256,256' --batch_size 1 --train_start --save_imgs --sample_distance 300 --beta_schedule 'linear' \
         --in_channels 3 --inference_num 4 --train_epochs 300 --model_save_freq 5 --inference_freq 10










[Screen 00]
python diffusion_training_vlb.py --device cuda:0 \
         --wandb_api_key '3a3bc2f629692fa154b9274a5bbe5881d47245dc' \
         --wandb_project_name 'MVTec_experiment' --wandb_run_name '0_no_kl_loss_one_step_infer' \
         --experiment_dir /data7/sooyeon/medical_image/MVTec_result/0_no_kl_loss_one_step_infer \
         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/MV/bagel/train/good/rgb' \
         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/MV/bagel/train/good/rgb' \
         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/MV/bagel/test/crack/rgb' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/MV/bagel/test/crack/rgb' \
         --img_siz '256,256' --batch_size 2 --train_start --save_imgs --sample_distance 50 --beta_schedule 'linear' \
         --in_channels 3 --inference_num 4 --train_epochs 300 --model_save_freq 5 --onestep_inference

[Screen 01]
python diffusion_training_vlb.py --device cuda:1 \
         --wandb_api_key '3a3bc2f629692fa154b9274a5bbe5881d47245dc' \
         --wandb_project_name 'MVTec_experiment' --wandb_run_name '1_no_kl_loss_step_infer' \
         --experiment_dir /data7/sooyeon/medical_image/MVTec_result/1_no_kl_loss_step_infer \
         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/MV/bagel/train/good/rgb' \
         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/MV/bagel/train/good/rgb' \
         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/MV/bagel/test/crack/rgb' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/MV/bagel/test/crack/rgb' \
         --img_siz '256,256' --batch_size 1 --train_start --save_imgs --sample_distance 300 --beta_schedule 'linear' \
         --in_channels 3 --inference_num 4 --train_epochs 300 --model_save_freq 5

[Screen 03]
python diffusion_training_vlb.py --device cuda:3 \
         --wandb_api_key '3a3bc2f629692fa154b9274a5bbe5881d47245dc' \
         --wandb_project_name 'MVTec_experiment' --wandb_run_name '2_kl_loss_one_step_infer' \
         --experiment_dir /data7/sooyeon/medical_image/MVTec_result/2_kl_loss_one_step_infer \
         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/MV/bagel/train/good/rgb' \
         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/MV/bagel/train/good/rgb' \
         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/MV/bagel/test/crack/rgb' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/MV/bagel/test/crack/rgb' \
         --img_siz '256,256' --batch_size 2 --train_start --save_imgs --sample_distance 300 --beta_schedule 'linear' \
         --in_channels 3 --inference_num 4 --train_epochs 300 --use_vlb_loss --onestep_inference --model_save_freq 5

[Screen 05]
python diffusion_training_vlb.py --device cuda:5 \
         --wandb_api_key '3a3bc2f629692fa154b9274a5bbe5881d47245dc' \
         --wandb_project_name 'MVTec_experiment' --wandb_run_name '3_kl_loss_step_infer' \
         --experiment_dir /data7/sooyeon/medical_image/MVTec_result/3_kl_loss_step_infer \
         --train_data_folder '/data7/sooyeon/medical_image/experiment_data/MV/bagel/train/good/rgb' \
         --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/MV/bagel/train/good/rgb' \
         --val_data_folder '/data7/sooyeon/medical_image/experiment_data/MV/bagel/test/crack/rgb' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/MV/bagel/test/crack/rgb' \
         --img_siz '256,256' --batch_size 2 --train_start --save_imgs --sample_distance 300 --beta_schedule 'linear' \
         --in_channels 3 --inference_num 4 --train_epochs 300 --use_vlb_loss --model_save_freq 5