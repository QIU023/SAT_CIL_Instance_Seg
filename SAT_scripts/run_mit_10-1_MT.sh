if  [ $# -ge 1 ] 
then
    DEVICES=$1
fi
CUDA_VISIBLE_DEVICES=${DEVICES} python3 initial_train.py --config yolact_mitb2_pascal_config_init_10with1 --batch_size 8 --save_folder weights/1-10/

