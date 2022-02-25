DEVICES=0

if  [ $# -ge 1 ] 
then
    DEVICES=$1
fi

exp_name='19-1_0_teacher_exp'
exp_name2='19-1_1_expert_exp'
exp_name3='19-1_1_MT_student_exp'


# CUDA_VISIBLE_DEVICES=${DEVICES} python3 initial_train.py --config yolact_mitb2_pascal_config_init --batch_size 8 --resume interrupt
# CUDA_VISIBLE_DEVICES=${DEVICES} nohup python3 expert_train.py \
#     --config yolact_mitb2_pascal_config_expert \
#     --batch_size 8 --num_workers 8 \
#     --save_folder weights/19-1 > 'train_log/'${exp_name2}'.log' 2>&1 &


    # --resume weights/19-1/new_model_mix_transformer.pth \

CUDA_VISIBLE_DEVICES=${DEVICES} nohup python3 incremental_train.py \
    --config yolact_mitb2_pascal_config_incremental \
    --batch_size 8 --num_workers 8 \
    --save_folder weights/VOC/19-1 \
    --load_expert_net weights/VOC/19-1/19-1_1_expert_final.pth \
    --load_distillation_net weights/VOC/19-1/19-1_0_teacher_final.pth > 'train_log/'${exp_name3}'.log' 2>&1 &

# CUDA_VISIBLE_DEVICES=${DEVICES} sh ./scripts/
# SATIS/weights/19-1/19-1_1_student_40497_interrupt.pth