python train_ldm.py --wandb_project_name 'dental_experiment' --wandb_run_name 'hand_10000_64res' \
                    --data_folder '/data7/sooyeon/medical_image/experiment_data/MedNIST/Hand' \
                    --autokl_training_epochs 100 \
                    --img_size '64,64' \
                    --experiment_basic_dir '/data7/sooyeon/medical_image/experiment_result/hand_10000_64res' \
                    --autoencoder_inference_num 5 \
                    --unet_training_epochs 300 \
                    --unet_val_interval 40 --device 'cuda:6'
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
python train_ldm.py --wandb_project_name 'dental_experiment' --wandb_run_name 'hand_1000_64res_high_epoch' \
                    --data_folder '/data7/sooyeon/medical_image/experiment_data/MedNIST/Hand_1000' \
                    --autokl_training_epochs 300 \
                    --img_size '64,64' \
                    --experiment_basic_dir '/data7/sooyeon/medical_image/experiment_result/hand_1000_64res_high_epoch' \
                    --autoencoder_inference_num 5 \
                    --unet_training_epochs 3000 \
                    --unet_val_interval 40
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
python train_ldm_org.py --wandb_project_name 'dental_experiment' \
                        --wandb_run_name '7_hand_10000_64_res_original_coder_check' \
                        --device 'cuda:4' \
                        --data_folder '/data7/sooyeon/medical_image/experiment_data/MedNIST/Hand' \
                        --use_original_autoencoder --use_pretrained_autoencoder \
                        --autoencoder_pretrained_dir '/data7/sooyeon/medical_image/experiment_result/4_hand_10000_original_code_64_res/vae_checkpoint_100.pth' \
                        --experiment_basic_dir '/data7/sooyeon/medical_image/experiment_result/7_hand_10000_64_res_original_coder_check' \
                        --autoencoder_inference_num 5 --unet_val_interval 10 --unet_training_epochs 300
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
python train_ldm_org.py --wandb_project_name 'dental_experiment' \
                        --wandb_run_name '10_dental_64_dentalai' \
                        --device 'cuda:6' \
                        --train_data_folder '/data7/sooyeon/medical_image/experiment_data/DentalAI/train_preprocessed' \
                        --val_data_folder '/data7/sooyeon/medical_image/experiment_data/DentalAI/test_preprocessed' \
                        --use_original_autoencoder --use_pretrained_autoencoder \
                        --autoencoder_pretrained_dir '/data7/sooyeon/medical_image/experiment_result/10_dental_64_dentalai/vae_checkpoint_500.pth' \
                        --autoencoder_training_epochs 500 \
                        --image_size 64 \
                        --experiment_basic_dir '/data7/sooyeon/medical_image/experiment_result/10_dental_64_dentalai' \
                        --autoencoder_inference_num 5 --unet_val_interval 10 --unet_training_epochs 300