if  [ $# -ge 1 ] 
then
    DEVICES=$1
fi

exp_name='10-1_0_student_exp'

CUDA_VISIBLE_DEVICES=${DEVICES} nohup python3 initial_train.py \
    --config yolact_mitb2_pascal_config_init_10with1 \
    --batch_size 8 \
    --save_folder weights/10-1/ \
    --resume weights/10-1/10-1_0_student.pth > 'train_log/'${exp_name}'.log' 2>&1 &

