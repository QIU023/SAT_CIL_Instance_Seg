DEVICES=4

# CUDA_VISIBLE_DEVICES=${DEVICES} python3 initial_train.py --config yolact_mitb2_pascal_config_init --batch_size 8 --resume interrupt
# CUDA_VISIBLE_DEVICES=${DEVICES} python3 expert_train.py --config yolact_mitb2_pascal_config_expert --batch_size 8 
CUDA_VISIBLE_DEVICES=${DEVICES} python3 incremental_train.py --config yolact_mitb2_pascal_config_incremental --resume interrupt --batch_size 8 --load_expert_net weights/20/mix_transformer_769_10000.pth --load_distillation_net weights/1-19/mix_transformer_114_30000.pth

# CUDA_VISIBLE_DEVICES=${DEVICES} sh ./scripts/
