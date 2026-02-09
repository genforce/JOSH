import multiprocessing as mp
import os
import subprocess
import sys
from argparse import ArgumentParser
from concurrent import futures
from glob import glob

GPUS = [0]


def run(input_folder, start_frame, log_file):
    cur_proc = mp.current_process()
    print("PROCESS", cur_proc.name, cur_proc._identity)
    worker_id = cur_proc._identity[0] - 1  # 1-indexed processes
    gpu = GPUS[worker_id % len(GPUS)]
    cmd = (f"CUDA_VISIBLE_DEVICES={gpu} "
           f"python josh/inference.py --input_folder {input_folder} --start_frame {start_frame} --num_frames 21")
    print(f"LOGGING TO {log_file}")
    cmd = f"{cmd} > {log_file} 2>&1"
    print(cmd)
    subprocess.call(cmd, shell=True)


def main(input_folder):
    log_dir = f"{input_folder}/logs"
    os.makedirs(log_dir, exist_ok=True)
    with futures.ProcessPoolExecutor(max_workers=len(GPUS)) as exe:
        num_frames = len(glob(f"{input_folder}/rgb/*.jpg"))
        start_frame = 0
        while num_frames - start_frame > 100:
            log_file = f"{log_dir}/{start_frame}.log"
            exe.submit(run, input_folder, start_frame, log_file)
            start_frame += 100


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_folder", type=str)
    args = parser.parse_args()
    main(args.input_folder)
