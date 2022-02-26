if  [ $# -ge 1 ] 
then
    DEVICES=$1
fi

exp_name='10-5_0_student_exp'
exp_name2='10-5_1_expert_exp'
exp_name3='10-5_1_MT_student_exp'
exp_name4='10-5_2_expert_exp'
exp_name5='10-5_2_MT_student_exp'

# CUDA_VISIBLE_DEVICES=${DEVICES} nohup python3 initial_train.py \
#     --config yolact_mitb2_pascal_config_init_10with1 \
#     --batch_size 8 \
#     --save_folder weights/10-1/ \
#     --resume weights/10-1/mix_transformer_76_47684_interrupt.pth > 'train_log/'${exp_name}'.log' 2>&1 &

# CUDA_VISIBLE_DEVICES=${DEVICES} nohup python3 expert_train.py \
#     --config MT_yolact_mitb2_pascal_config_incremental_10with5_1_expert \
#     --batch_size 8 --num_workers 8 \
#     --save_folder weights/VOC/10-5 > 'train_log/'${exp_name2}'.log' 2>&1 &

# CUDA_VISIBLE_DEVICES=${DEVICES} nohup python3 expert_train.py \
#     --config MT_yolact_mitb2_pascal_config_incremental_10with5_2_expert \
#     --batch_size 8 --num_workers 8 \
#     --save_folder weights/VOC/10-5_2_expert_tempdir > 'train_log/'${exp_name4}'.log' 2>&1 &

CUDA_VISIBLE_DEVICES=${DEVICES} nohup python3 incremental_train.py \
    --config MT_yolact_mitb2_pascal_config_incremental_10with5_1_incremental \
    --batch_size 8 --num_workers 8 \
    --load_distillation_net weights/VOC/10-5/10-5_0_teacher_final.pth \
    --load_expert_net weights/VOC/10-5/10-5_1_expert_final.pth \
    --save_folder weights/VOC/10-5_tempdir/ > 'train_log/'${exp_name3}'.log' 2>&1 &
