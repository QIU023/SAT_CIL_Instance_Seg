if  [ $# -ge 1 ] 
then
    DEVICES=$1
fi

exp_name='5-3_0_student_exp'

# CUDA_VISIBLE_DEVICES=${DEVICES} nohup python3 initial_train.py \
#     --config yolact_mitb2_pascal_config_init_5with3 \
#     --batch_size 8 \
#     --save_folder weights/5-3/ > 'train_log/'${exp_name}'.log' 2>&1 &

exp_name2='5-3_1_expert_exp'

exp_name3='5-3_1_SAT_exp'

CUDA_VISIBLE_DEVICES=${DEVICES} nohup python3 incremental_train.py \
    --config SAT_yolact_mitb2_pascal_config_incremental_5with3_1 \
    --batch_size 8 \
    --load_distillation_net weights/5-3/5-3_0_student_final.pth \
    --save_folder weights/5-3/ > 'train_log/'${exp_name3}'.log' 2>&1 &

# CUDA_VISIBLE_DEVICES=${DEVICES} python3 initial_train.py \
#     --config yolact_mitb2_pascal_config_init_15with5 \
#     --batch_size 8 \
#     --save_folder weights/5-3/

# CUDA_VISIBLE_DEVICES=${DEVICES} nohup python3 expert_train.py \
#     --config yolact_mitb2_pascal_config_expert_5with3 \
#     --batch_size 8  --task 5-3 --step 1 \
#     --save_folder weights/5-3  > 'train_log/'${exp_name2}'.log' 2>&1 &