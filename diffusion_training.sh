python diffusion_training.py --experiment_dir data7/sooyeon/medical_image/anoddpm_result/test --device cuda:4 \
                             --wandb_project_name 'dental_experiment_anoddpm' --wandb_run_name 'dental_256res' \
                             --train_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_256' \
                             --val_data_folder '/data7/sooyeon/medical_image/experiment_data/dental/panoramic_data_res_256' \
                             --img_siz '256,256' --batch_size 32





  "EPOCHS": 3000,
  "T": 1000,
  "base_channels": 128,
  "beta_schedule": "linear",
  "channel_mults": "",
  "loss-type": "l2",
  "loss_weight": "none",
  "train_start": true,
  "lr": 1e-4,
  "random_slice": true,
  "sample_distance": 800,
  "weight_decay": 0.0,
  "save_imgs": false,
  "save_vids": true,
  "dropout": 0,
  "attention_resolutions": "16,8",
  "num_heads": 2,
  "num_head_channels": -1,
  "noise_fn": "simplex",
  "dataset": "mri"
}                             "