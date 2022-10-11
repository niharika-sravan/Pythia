import subprocess
import time

jobstr = '''#!/bin/bash
#SBATCH --job-name="{0}"
#SBATCH --output="temp/{0}.%j.%N.out"
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --mem=249325M
#SBATCH --account=TG-AST200029
#SBATCH --export=ALL
#SBATCH --constraint="lustre"
#SBATCH -t 48:00:00

module load cpu
module load parallel/20200822
source /home/niharika/.bash_profile
conda activate kne
cd /expanse/lustre/projects/umn131/niharika/Pythia

'''

alpha_list = [1e-0, 1e-1, 1e-2]
gamma_list = [0.9, 0.5, 0.1]
n_list = [2.]#, 3.]

for i, alpha in enumerate(alpha_list):
  for j, gamma in enumerate(gamma_list):
    for k, n in enumerate(n_list):
      #cmd = 'python train.py '+str(alpha)+' '+str(gamma)+' '+str(n)+' random'
      cmd = 'python train.py '+str(alpha)+' '+str(gamma)+' '+str(n)+' outdir/n9/rand/train_linw_0.1_0.9_2.0_461.npy'
      job = 'train_'+str(alpha)+'_'+str(gamma)+'_'+str(n)
      with open('temp/'+job+'.sub', 'w') as f:
        f.writelines(jobstr.format(job)+cmd)
      proc = subprocess.Popen(['sbatch', 'temp/'+job+'.sub'])
      time.sleep(0.2)

'''
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=9
#SBATCH --gpus=1
#SBATCH --mem=90G
'''
