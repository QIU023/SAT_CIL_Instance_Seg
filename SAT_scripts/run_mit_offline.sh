if  [ $# -ge 1 ] 
then
    DEVICES=$1
fi

exp_name='offline_exp'

# CUDA_VISIBLE_DEVICES=${DEVICES} python3 initial_train.py --config yolact_mitb2_pascal_config_init --batch_size 8 --resume interrupt
# CUDA_VISIBLE_DEVICES=${DEVICES} python3 expert_train.py --config yolact_mitb2_pascal_config_expert --batch_size 8 

CUDA_VISIBLE_DEVICES=${DEVICES} nohup python3 initial_train.py \
    --config yolact_mitb2_pascal_config_offline \
    --batch_size 8 \
    --save_folder weights/offline  > 'train_log/'${exp_name}'.log' 2>&1 &

# CUDA_VISIBLE_DEVICES=${DEVICES} sh ./scripts/
# SATIS/weights/19-1/19-1_1_student_40497_interrupt.pth