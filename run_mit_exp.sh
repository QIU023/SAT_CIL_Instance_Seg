DEVICES=1,2,6,7

# CUDA_VISIBLE_DEVICES=${DEVICES} python3 initial_train.py --config yolact_mitb2_pascal_config_init --batch_size 32 --resume interrupt
CUDA_VISIBLE_DEVICES=${DEVICES} python3 expert_train.py --config yolact_mitb2_pascal_config_expert --batch_size 32 
# CUDA_VISIBLE_DEVICES=${DEVICES} python3 incremental_train.py --config yolact_mitb2_pascal_config_incremental --batch_size 32 --load_distillation_net /opt/tiger/occupy_arnold/SATIS/weights/1-19/mix_transformer_114_30000.pth

# CUDA_VISIBLE_DEVICES=${DEVICES} sh ./scripts/
