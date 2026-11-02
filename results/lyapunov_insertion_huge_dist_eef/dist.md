# 我在命令行输入的干扰是这样的：

python eval_lyapunov.py   --gpu 0   --ckpt_dir "/mnt/e9933268-1ca5-423d-88b2-ec7b2e67a577/Coding/Viper_l1loss_0padding_450train_450eval/ckpt/sim_insertion_scripted/Viper_chunksize_50_stc_bs_16"   --task_name sim_insertion_scripted   --dataset_dir "/mnt/e9933268-1ca5-423d-88b2-ec7b2e67a577/Coding/Viper_l1loss_0padding_450train_450eval/data_450_lyapunov_ref/sim_insertion_scripted"   --episode_id 0   --disturb_t 60   --disturb_type eef   --delta_left_arm 1 1 1 1 0 0   --delta_right_arm 0 1 1 1 1 0   --results_dir results/lyapunov_insertion_huge_dist_eef    --save_video  --video_path results/lyapunov_insertion_huge_dist_eef/my_rollout.mp4


# 然后绘图使用下述命令：

python other_utils/plot_lyapunov_figure.py   --npz results/lyapunov_insertion_huge_dist_eef/lyapunov_ep0_dist60_eef.npz   --meta_json results/lyapun
ov_insertion_huge_dist_eef/lyapunov_ep0_dist60_eef_meta.json   --out results/lyapunov_i
nsertion_huge_dist_eef/fig_lyapunov.png