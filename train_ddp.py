import os
import subprocess
import numpy as np
from time import sleep
import sys
# from configs.config_mt_gpt_ddp2 import *
from configs.config_mt_ddp import *
import signal

procs = []
for i in range(num_gpus):
    env_copy = os.environ.copy()
    env_copy["WORLD_SIZE"] = str(num_gpus)
    env_copy["NODE_RANK"] = str(i)
    scripts = [sys.executable, "run_ddp.py"]
    proc = subprocess.Popen(scripts, env=env_copy)
    procs.append(proc)
    delay = np.random.uniform(1, 3, 1)[0]
    sleep(delay)

for proc in procs:
    try:
        proc.wait()
    except KeyboardInterrupt:
        try:
            # Send CTRL+c to kill the child process from su -
            proc.send_signal(signal.SIGINT)
        except subprocess.TimeoutExpired:
            print('Timeout killing subprocess')
            proc.kill()
