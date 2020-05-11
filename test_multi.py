import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--default')
# Required parameters
parser.add_argument(
    "--model_type",
    default=None,
    type=str,
    required=True,
)   
parser.add_argument(
    "--model_name_or_path",
    action='append',
)   
parser.add_argument(
    "--model_name",
    default=None,
    type=str,
    help="The model name of the pre-trained model",
)   
parser.add_argument(
    "--output_dir",
    default=None,
    type=str,
    required=True,
    help="The output directory where the model checkpoints will be written.",
)   
#parser.add_argument(
#    "--output_result_dir",
#    default=None,
#    type=str,
#    required=True,
#    help="The output directory where the model predictions will be written.",
#)   
parser.add_argument(
    "--data_dir",
    default=None,
    type=str,
    help="The input data dir. Should contain the .json files for the task."
    + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
)
parser.add_argument(
    "--predict_file",
    default=None,
    type=str,
    help="The input evaluation file. If a data dir is specified, will look for the file there"
    + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
)
parser.add_argument(
    "--max_answer_length",
    default=30,
    type=int,
    help="The maximum length of an answer that can be generated. This is needed because the start "
    "and end predictions are not conditioned on one another.",
)
parser.add_argument(
    "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
)
parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")

args = parser.parse_args()
code_dir =  os.getcwd()
code_path = os.path.join(code_dir,"run.py")

cmd_base = "python {} --data_dir {} --model_name_or_path {} --output_dir {} --model_type {} --predict_file {} --do_eval --overwrite_output_dir --max_seq_length 256 --per_gpu_eval_batch_size {} --n_best_size {} --split_doc --warmup_proportion 0.1 --seed 12345 --weight_decay 0.01 --adam_epsilon 1e-6 --do_lower_case --gradient_accumulation_steps 16 --threads {} --model_name {} --output_all_logit --max_answer_length {}  --output_result_dir {}"


for path in args.model_name_or_path:
    cmd = cmd_base.format( code_path,
                args.data_dir, path, args.output_dir, args.model_type, args.predict_file,
                args.per_gpu_eval_batch_size, args.n_best_size, args.threads, args.model_name, args.max_answer_length, 
                path)
    #print (cmd)
    os.system(cmd)
