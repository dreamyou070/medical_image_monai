device='cuda:3'
experiment_dir='/data7/sooyeon/medical_image/anoddpm_result/7_gaussian_linear_pos_infonce'

python generate_heatmap.py --device=${device} \
                           --experiment_dir=${experiment_dir}