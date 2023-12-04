python diffusion_inference.py --device 'cuda:0' --process_title 'parksooyeon' \
   --wandb_project_name 'inference_test' --wandb_run_name 'dental_inference_step_check' \
   --experiment_dir /data7/sooyeon/medical_image/MVTec_result/2_kl_loss \
   --train_data_folder '/data7/sooyeon/medical_image/experiment_data/MV/bagel/train/good/rgb' \
   --train_mask_dir '/data7/sooyeon/medical_image/experiment_data/MV/bagel/train/good/rgb' \
   --val_data_folder '/data7/sooyeon/medical_image/experiment_data/MV/bagel/test/crack/rgb' \
   --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/MV/bagel/test/crack/rgb' \
   --unet_pretrained_dir '/data7/sooyeon/medical_image/MVTec_result/2_kl_loss/diffusion-models/unet_epoch_20.pt' \
   --in_channels 3 --img_siz '256,256' --batch_size 1 --train_start --save_imgs --sample_distance 300 --beta_schedule 'linear' \