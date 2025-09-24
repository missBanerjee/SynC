#PYTHON='/home/duruoyi/anaconda3/envs/py38/bin/python3.8'

hostname
nvidia-smi

#export CUDA_VISIBLE_DEVICES=0

# Get unique log file,
SAVE_DIR=/home/anwesha/On_the_fly/On-the-fly-Category-Discovery/OCD/nearest_cls_align/proto/arachnida/
mkdir -p $SAVE_DIR
EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

python -m fine \
            --dataset_name 'Arachnida' \
            --batch_size 128 \
            --grad_from_block 11 \
            --epochs 10\
            --k 3.5\
            --base_model vit_dino \
            --num_workers 16 \
            --use_ssb_splits 'False' \
            --sup_con_weight 0.5 \
            --warmup_model_dir '/home/anwesha/On_the_fly/On-the-fly-Category-Discovery/dev_outputs/checkpoints/log/(20.07.2025_|_50.745)/checkpoints/model_best.pt'\
            --warmup_proj_dir '/home/anwesha/On_the_fly/On-the-fly-Category-Discovery/dev_outputs/checkpoints/log/(20.07.2025_|_50.745)/checkpoints/model_proj_head_best.pt'\
            --weight_decay 5e-5 \
            --contrast_unlabel_only 'False' \
            --transform 'imagenet' \
            --lr 0.001\
            --eval_funcs 'v2' \
> ${SAVE_DIR}logfile_${EXP_NUM}.out