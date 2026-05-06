#!/bin/bash
dataset_list=('ABIDE')
batch_size_list=(64 32 16)
base_lr_list=(1e-5)
target_lr_list=(1e-4)
wd_list=(1e-3 1e-4 1e-5)
activation_list=('leaky_relu')
dropout_list=(0.1 0.2 0.3)
n_clusters_list=('[128,64]' '[64,32]' '[128,32]' '[128,64,32]' '[64,32,16]' '[128]' '[64]' '[32]')
alpha_list=(0.0 0.05 0.1 0.2 0.4 0.8)
s_mlp_layer_list=(1 2 3)
sc_heads_list=(1 2 4 8 16)



for name in "${dataset_list[@]}"; do
  for acl in "${activation_list[@]}"; do
    for b_lrl in "${base_lr_list[@]}"; do
      for t_lrl in "${target_lr_list[@]}"; do
        for wdl in "${wd_list[@]}"; do
          for bzl in "${batch_size_list[@]}"; do
            for dl in "${dropout_list[@]}"; do
              for clus_numl in "${n_clusters_list[@]}"; do
                for alpha in "${alpha_list[@]}"; do
                  for s_mlp_layer in "${s_mlp_layer_list[@]}"; do
                    for sc_heads in "${sc_heads_list[@]}"; do
                        python main.py --dataset $name \
                                    --batch_size $bzl \
                                    --base_lr $b_lrl \
                                    --target_lr $t_lrl \
                                    --weight_decay $wdl \
                                    --activation $acl \
                                    --dropout $dl \
                                    --n_clusters $clus_numl \
                                    --alpha $alpha \
                                    --s_mlp_layer $s_mlp_layer \
                                    --sc_heads $sc_heads
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done


