if  [ $# -ge 1 ] 
then
    DEVICES=$1
fi

exp_name='10-1_0_student_exp'
exp_name4='10-5_1_SAT_exp'
exp_name5='10-5_2_SAT_exp'
# CUDA_VISIBLE_DEVICES=${DEVICES} nohup python3 initial_train.py \
#     --config yolact_mitb2_pascal_config_init_10with1 \
#     --batch_size 8 \
#     --save_folder weights/10-1/ \
#     --resume weights/10-1/mix_transformer_76_47684_interrupt.pth > 'train_log/'${exp_name}'.log' 2>&1 &

CUDA_VISIBLE_DEVICES=${DEVICES} nohup python3 incremental_train.py \
    --config SAT_yolact_mitb2_pascal_config_incremental_10with5_1 \
    --batch_size 8 \
    --load_distillation_net weights/10-5/10-5_0_student_final.pth \
    --save_folder weights/15-5/ > 'train_log/'${exp_name4}'.log' 2>&1 &

# CUDA_VISIBLE_DEVICES=${DEVICES} nohup python3 incremental_train.py \
#     --config SAT_yolact_mitb2_pascal_config_incremental_10with5_2 \
#     --batch_size 8 \
#     --load_distillation_net weights/10-5/10-5_1_student_final.pth \
#     --save_folder weights/15-5/ > 'train_log/'${exp_name5}'.log' 2>&1 &