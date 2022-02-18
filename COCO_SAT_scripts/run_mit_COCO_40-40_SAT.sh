DEVICES=0

# CUDA_VISIBLE_DEVICES=${DEVICES} python3 initial_train.py --config yolact_mitb2_pascal_config_init_15with5 --batch_size 8 --save_folder weights/15-5/

if  [ $# -ge 1 ] 
then
    DEVICES=$1
fi

exp_name1='coco_40-40_0_student_exp'
exp_name2='coco_40-40_1_expert_exp'
# CUDA_VISIBLE_DEVICES=${DEVICES} nohup python3 initial_train.py \
#     --config yolact_mitb2_coco_config_40with40 \
#     --batch_size 8 --num_workers 8 \
#     --save_folder weights/COCO/40-40 > 'train_log/'${exp_name1}'.log' 2>&1 &

CUDA_VISIBLE_DEVICES=${DEVICES} nohup python3 expert_train.py \
    --config yolact_mitb2_coco_config_40with40_expert \
    --batch_size 8 --num_workers 8 \
    --save_folder weights/COCO/40-40-expert  > 'train_log/'${exp_name2}'.log' 2>&1 &

# CUDA_VISIBLE_DEVICES=${DEVICES} nohup python3 incremental_train.py \
#     --config SAT_yolact_mitb2_pascal_config_incremental_15with5 \
#     --batch_size 8 --num_workers 8 \
#     --load_distillation_net weights/15-5/15-5_0_student_final.pth \
#     --save_folder weights/15-5/ > 'train_log/'${exp_name4}'.log' 2>&1 &

# CUDA_VISIBLE_DEVICES=${DEVICES} sh ./scripts/
