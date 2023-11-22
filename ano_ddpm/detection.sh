python detection.py --device 'cuda:6' --noise_fn "gauss" \
                    --unet_state_dict_dir "/data7/sooyeon/medical_image/anoddpm_result/1_only_normal_training_gaussian/diffusion-models/unet_epoch_100.pt" \
                    --dataset_path '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original'