DEVICES=0,1,2,3

CUDA_VISIBLE_DEVICES=${DEVICES} python3 initial_train.py --config yolact_mitb2_pascal_config_init_15with5 --batch_size 32 --save_folder weights/5-15/
# CUDA_VISIBLE_DEVICES=${DEVICES} python3 expert_train.py --config yolact_mitb2_pascal_config_expert_15with5 --batch_size 32  --save_folder weights/16to20/
# CUDA_VISIBLE_DEVICES=${DEVICES} python3 incremental_train.py --config yolact_mitb2_pascal_config_incremental_15with5 --batch_size 32 --load_distillation_net weights/1-19/mix_transformer_114_30000.pth  --save_folder weights/15+5/

# CUDA_VISIBLE_DEVICES=${DEVICES} sh ./scripts/
