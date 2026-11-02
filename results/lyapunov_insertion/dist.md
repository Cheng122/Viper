python eval_lyapunov.py   --gpu 0   --ckpt_dir "/mnt/e9933268-1ca5-423d-88b2-ec7b2e67a577/Coding/Viper_l1loss_0padding_450train_450eval/ckpt/sim_insertion_scripted/Viper_chunksize_50_stc_bs_16"   --task_name sim_insertion_scripted   --dataset_dir "/mnt/e9933268-1ca5-423d-88b2-ec7b2e67a577/Coding/Viper_l1loss_0padding_450train_450eval/data_450_lyapunov_ref/sim_insertion_scripted"   --episode_id 0   --results_dir results/lyapunov_insertion   --lyapunov_log_stride 10   --omega_pro 1.0 --omega_img 1.0 --lam_u 1.0


python other_utils/plot_lyapunov_figure.py \
  --npz results/lyapunov_insertion/lyapunov_ep0.npz \
  --meta_json results/lyapunov_insertion/lyapunov_ep0_meta.json \
  --out results/lyapunov_insertion/fig_lyapunov.png


python other_utils/plot_lyapunov_compare.py \
  --out results/lyapunov_insertion/compare_four.png \
  --series 'Nominal (no perturbation)|results/lyapunov_insertion/lyapunov_ep0.npz' \
  --series 'Object translation ($n_d=60$)|results/lyapunov_insertion_dist_obs/lyapunov_ep0_dist60_object.npz' \
  --series 'Large EEF joint bias ($n_d=60$)|results/lyapunov_insertion_huge_dist_eef/lyapunov_ep0_dist60_eef.npz' \
  --series 'EEF joint bias ($n_d=90$)|results/lyapunov_insertion_dist_eef/lyapunov_ep0_dist90_eef.npz'