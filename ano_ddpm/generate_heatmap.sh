model_name='unet_epoch_250.pt'
experiment_dir='/data7/sooyeon/medical_image/anoddpm_result/7_gaussian_linear_pos_infonce'
img_size='128,128'
train_data_folder='/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/original'
train_mask_dir='/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/train/mask'
val_data_folder='/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/original'
val_mask_dir='/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_128/valid/mask'
batch_size=10
thredhold=0.0005

python generate_heatmap.py --device 'cuda:1' \
                           --experiment_dir=${experiment_dir} \
                           --model_name=${model_name} \
                           --img_size=${img_size} \
                           --batch_size=${batch_size} \
                           --train_data_folder=${train_data_folder} \
                           --train_mask_dir=${train_mask_dir} \
                           --val_data_folder=${val_data_folder} \
                           --val_mask_dir=${val_mask_dir} \
                           --thredhold=${thredhold}