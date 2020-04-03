source activate torch
export CUDA_VISIBLE_DEVICES=0
python run.py --data_dir /data/bzw/MRC/data/lic2020/dureader_robust-data \
    --model_name_or_path  albert_models/checkpoint-9805 \
    --output_dir ./ \
    --model_type albert \
    --predict_file dev.json \
    --do_eval \
    --overwrite_output_dir \
    --max_seq_length 256 \
    --per_gpu_eval_batch_size 32 \
#    --evaluate_during_training \
    
