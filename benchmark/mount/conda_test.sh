
# check if conda is initialized
if [ -z "$CONDA_EXE" ]; then
    echo "Conda not initialized"
    conda init bash
fi

# source ~/.bashrc
source /opt/conda/bin/activate
conda activate /opt/ad_env

conda info --envs

python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.compile)"
