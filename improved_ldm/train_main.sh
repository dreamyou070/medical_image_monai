python train_main.py --data_dir '/data7/sooyeon/medical_image/experiment_data/MV/bagel/train/good/rgb' \
         --mask_dir '/data7/sooyeon/medical_image/experiment_data/MV/bagel/train/good/rgb' \
         --val_data_dir '/data7/sooyeon/medical_image/experiment_data/MV/bagel/test/crack/rgb' \
         --val_mask_dir '/data7/sooyeon/medical_image/experiment_data/MV/bagel/test/crack/rgb' \
         --img_siz '256,256' --batch_size 1 --train_start --save_imgs --sample_distance 300 --beta_schedule 'linear' \
         --in_channels 3 --inference_num 4 --train_epochs 300 --model_save_freq 5 --onestep_inference