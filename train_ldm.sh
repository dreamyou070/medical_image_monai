python train_ldm.py --wandb_project_name 'dental_experiment' --wandb_run_name 'hand_1000_64res' \
                    --data_folder '/data7/sooyeon/medical_image/experiment_data/MedNIST/Hand_1000' \
                    --autokl_training_epochs 100 \
                    --image_size '64,64' \
                    --experiment_basic_dir '/data7/sooyeon/medical_image/experiment_result/hand_1000_64res' \
                    --autoencoder_inference_num 5 \
                    --unet_training_epochs 300 \
                    --unet_val_interval 40