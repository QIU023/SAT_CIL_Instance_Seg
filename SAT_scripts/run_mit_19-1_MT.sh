if  [ $# -ge 1 ] 
then
    DEVICES=$1
fi

exp_name='19-1_1_student_exp'

# CUDA_VISIBLE_DEVICES=${DEVICES} python3 initial_train.py --config yolact_mitb2_pascal_config_init --batch_size 8 --resume interrupt
# CUDA_VISIBLE_DEVICES=${DEVICES} python3 expert_train.py --config yolact_mitb2_pascal_config_expert --batch_size 8 

CUDA_VISIBLE_DEVICES=${DEVICES} nohup python3 incremental_train.py \
    --config yolact_mitb2_pascal_config_incremental \
    --resume weights/19-1/mix_transformer_2264_120000.pth \
    --batch_size 8 \
    --save_folder weights/19-1 \
    --load_expert_net weights/19-1/19-1_20expect_network.pth \
    --load_distillation_net weights/19-1/19-1_0_old_teacher.pth > 'train_log/'${exp_name}'.log' 2>&1 &

# CUDA_VISIBLE_DEVICES=${DEVICES} sh ./scripts/
# SATIS/weights/19-1/19-1_1_student_40497_interrupt.pth