#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --job-name=inversion
#SBATCH --ntasks=1
#SBATCH --partition=A2V80_ICE56M256_L
#SBATCH -e err.log
#SBATCH -o out.log
#SBATCH -A gol_yxi251_uksr
#SBATCH --export=NONE

unset LD_LIBRARY_PATH

module load ccs/Miniconda3
module load ccs/singularity

IMG=/share/singularity/images/ccs/conda/lcc-jupyter-rocky8.sinf

echo "---- Running inside container ----"
singularity exec --nv "$IMG" bash -lc '
  echo "Container OS: $(cat /etc/os-release | grep PRETTY_NAME)"
  uv run -m ppo.evaluate_temporal --config config/eval_temporal.yaml
  echo "---- Container execution completed ----"
'
