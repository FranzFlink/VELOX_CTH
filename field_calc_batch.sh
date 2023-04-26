#!/bin/bash

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate jomueller_env
cd ~/VELOX_CTH
python CTH_calculate_field.py --$1
conda deactivate