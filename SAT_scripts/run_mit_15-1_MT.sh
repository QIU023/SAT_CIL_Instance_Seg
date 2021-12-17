if  [ $# -ge 1 ] 
then
    DEVICES=$1
fi

exp_name='15-1_0_student_exp'

CUDA_VISIBLE_DEVICES=${DEVICES} nohup python3 initial_train.py \
    --config yolact_mitb2_pascal_config_init_15with5 \
    --batch_size 8 \
    --save_folder weights/15-1 \
    --resume weights/15-5/mix_transformer_7_7053_interrupt.pth
    --save_folder weights/15-5/  > 'train_log/'${exp_name}'.log' 2>&1 &
# CUDA_VISIBLE_DEVICES=${DEVICES} python3 expert_train.py --config yolact_mitb2_pascal_config_expert_15with5 --batch_size 32  --save_folder weights/16to20/
# CUDA_VISIBLE_DEVICES=${DEVICES} python3 incremental_train.py --config yolact_mitb2_pascal_config_incremental_15with5 --batch_size 32 --load_distillation_net weights/1-19/mix_transformer_114_30000.pth  --save_folder weights/15+5/

# CUDA_VISIBLE_DEVICES=${DEVICES} sh ./scripts/
