if  [ $# -ge 1 ] 
then
    DEVICES=$1
fi

exp_name='resnet_15-1_0_student_exp'
exp_name2='resnet_15-5_1_expert_exp'

# CUDA_VISIBLE_DEVICES=${DEVICES} nohup python3 initial_train.py \
#     --config yolact_resnet50_pascal_config_init_15with5 \
#     --batch_size 8 \
#     --save_folder weights/15-5/ > 'train_log/'${exp_name}'.log' 2>&1 &

CUDA_VISIBLE_DEVICES=${DEVICES} nohup python3 expert_train.py \
    --config yolact_resnet50_pascal_config_expert_15with5 \
    --step 1 --batch_size 8 --no_log \
    --save_folder weights/15-5/ > 'train_log/'${exp_name}'.log' 2>&1 &

# CUDA_VISIBLE_DEVICES=${DEVICES} python3 incremental_train.py --config yolact_mitb2_pascal_config_incremental_15with5 --batch_size 32 --load_distillation_net weights/1-19/mix_transformer_114_30000.pth  --save_folder weights/15+5/

# CUDA_VISIBLE_DEVICES=${DEVICES} sh ./scripts/
