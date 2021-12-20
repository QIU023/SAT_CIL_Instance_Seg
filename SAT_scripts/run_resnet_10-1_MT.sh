if  [ $# -ge 1 ] 
then
    DEVICES=$1
fi

exp_name='resnet_10-1_0_student_exp'

CUDA_VISIBLE_DEVICES=${DEVICES} nohup python3 initial_train.py \
    --config yolact_resnet50_pascal_config_init_10with1 \
    --batch_size 8 \
    --save_folder weights/10-1/ > 'train_log/'${exp_name}'.log' 2>&1 &
# CUDA_VISIBLE_DEVICES=${DEVICES} python3 expert_train.py --config yolact_mitb2_pascal_config_expert_15with5 --batch_size 32  --save_folder weights/16to20/
# CUDA_VISIBLE_DEVICES=${DEVICES} python3 incremental_train.py --config yolact_mitb2_pascal_config_incremental_15with5 --batch_size 32 --load_distillation_net weights/1-19/mix_transformer_114_30000.pth  --save_folder weights/15+5/

# CUDA_VISIBLE_DEVICES=${DEVICES} sh ./scripts/
