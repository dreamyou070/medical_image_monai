python autoencoder_training_idea.py --wandb_project_name 'dental_test' \
                                    --wandb_run_name 'four_advarsarial_loss_comparing_check' \
                                    --save_basic_dir '../experiment_result/four_advarsarial_loss_comparing_check' \
                                    --val_interval 20 --model_save_num 90 --n_epochs 500 --image_size '64,64' --device cuda:7 --batch_size 16 \
                                    --data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data'