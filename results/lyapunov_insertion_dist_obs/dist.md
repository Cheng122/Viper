python eval_lyapunov.py   --gpu 0   --ckpt_dir "/mnt/e9933268-1ca5-423d-88b2-ec7b2e67a577/Coding/Viper_l1loss_0padding_450train_450eval/ckpt/sim_insertion_scripted/Viper_chunksize_50_stc_bs_16"   --task_name sim_insertion_scripted   --dataset_dir "/mnt/e9933268-1ca5-423d-88b2-ec7b2e67a577/Coding/Viper_l1loss_0padding_450train_450eval/data_450_lyapunov_ref/sim_insertion_scripted"   --episode_id 0   --disturb_t 60   --disturb_type object   --delta_xyz 0.05 0.05 0.0   --results_dir results/lyapunov_insertion_dist_obs  --save_video  --video_path results/lyapunov_insertion_dist_obs/my_rollout.mp4


python other_utils/plot_lyapunov_figure.py   --npz results/lyapunov_insertion_dist_obs/lyapunov_ep0_dist60_object.npz   --meta_json results/lyapunov_insertion_dist_obs/lyapunov_ep0_dist60_object_meta.json   --out results/lyapunov_inser
tion_dist_obs/fig_lyapunov.png
Saved results/lyapunov_insertion_dist_obs/fig_lyapunov.png