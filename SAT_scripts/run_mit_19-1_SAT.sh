DEVICES=0

if  [ $# -ge 1 ] 
then
    DEVICES=$1
fi

exp_name='19-1_1_SAT_exp'

# CUDA_VISIBLE_DEVICES=${DEVICES} nohup python3 initial_train.py \
#     --config yolact_mitb2_pascal_config_init \
#     --batch_size 8 --num_workers 8 \
#     --save_folder weights/19-1/ \
#     --resume weights/19-1/19_1_0_teacher_final.pth > 'train_log/'${exp_name}'.log' 2>&1 &

CUDA_VISIBLE_DEVICES=${DEVICES} nohup python3 incremental_train.py \
    --config SAT_yolact_mitb2_pascal_config_incremental_19with1 \
    --batch_size 8 --num_workers 8 \
    --load_distillation_net weights/VOC/19-1/19_1_0_teacher_final.pth \
    --load_expert_net weights/VOC/19-1/19-1_1_expert_final.pth \
    --save_folder weights/VOC/19-1/ > 'train_log/'${exp_name}_1'.log' 2>&1 &
    
    # --resume weights/19-1/new_model_mix_transformer.pth \

# CUDA_VISIBLE_DEVICES=${DEVICES} nohup python3 incremental_train.py \
#     --config SAT_yolact_mitb2_pascal_config_incremental_10with5_2 \
#     --batch_size 8 \
#     --load_distillation_net weights/10-5/10-5_1_student_final.pth \
#     --save_folder weights/15-5/ > 'train_log/'${exp_name5}'.log' 2>&1 &