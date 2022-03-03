DEVICES=0

if  [ $# -ge 1 ] 
then
    DEVICES=$1
fi

exp_name='10-5_0_student_exp'
exp_name4='10-5_1_SAT_exp'
exp_name5='10-5_2_SAT_exp'
# CUDA_VISIBLE_DEVICES=${DEVICES} nohup python3 initial_train.py \
#     --config yolact_mitb2_pascal_config_init_10with1 \
#     --batch_size 8 --num_workers 8 \
#     --save_folder weights/10-5/ \
#     --resume weights/10-5/10-5_0_student.pth > 'train_log/'${exp_name}'.log' 2>&1 &

# CUDA_VISIBLE_DEVICES=${DEVICES} nohup python3 incremental_train.py \
#     --config SAT_yolact_mitb2_pascal_config_incremental_10with5_1 \
#     --batch_size 8 --num_workers 8 \
#     --load_distillation_net weights/10-5/10-5_0_student.pth \
#     --resume weights/10-5_1_SAT_student_final.pth \
#     --save_folder weights/10-5/ > 'train_log/'${exp_name4}'.log' 2>&1 &

# CUDA_VISIBLE_DEVICES=${DEVICES} nohup python3 expert_train.py \
#     --config yolact_mitb2_pascal_config_expert_15with5 \
#     --batch_size 8 --num_workers 8 \
#     --save_folder weights/15-5 > 'train_log/'${exp_name2}'.log' 2>&1 &

# CUDA_VISIBLE_DEVICES=${DEVICES} nohup python3 incremental_train.py \
#     --config SAT_yolact_mitb2_pascal_config_incremental_10with5_2 \
#     --batch_size 8 \
#     --load_distillation_net weights/10-5/10-5_1_student_final.pth \
#     --save_folder weights/10-5/ > 'train_log/'${exp_name5}'.log' 2>&1 &


CUDA_VISIBLE_DEVICES=${DEVICES} nohup python3 incremental_train.py \
    --config SAT_yolact_mitb2_pascal_config_incremental_10with5_2 \
    --batch_size 8 --num_workers 8 \
    --load_distillation_net weights/VOC/10-5/10-5_1_SAT_student_final.pth \
    --load_expert_net weights/VOC/10-5/10-5_2_expert_final.pth \
    --save_folder weights/VOC/10-5/ > 'train_log/'${exp_name5}'.log' 2>&1 &