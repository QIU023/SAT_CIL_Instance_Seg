DEVICES=0,1,2,3

CUDA_VISIBLE_DEVICES=${DEVICES} python3 initial_train.py --config yolact_resnet50_pascal_config_init --batch_size 32
# CUDA_VISIBLE_DEVICES=${DEVICES} python3 incremental_train.py --config yolact_resnet50_pascal_config_init --batch_size 32

# CUDA_VISIBLE_DEVICES=${DEVICES} sh ./scripts/
