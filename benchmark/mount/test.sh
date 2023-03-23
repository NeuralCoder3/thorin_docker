FOLDERS="10k_small 10k 10k_D128 10k_K100 10k_D256 10k_K200"
# RESULT_DIR="results/gmm"
RESULT_DIR="/home/s8maullr/results"

EXEC_DIR=/etc/ad/mount

cd $EXEC_DIR

for FOLDER in $FOLDERS; do
    echo "Executing GMM $FOLDER benchmark"
    OUTPUT_DIR="$RESULT_DIR/$FOLDER"
    mkdir -p $OUTPUT_DIR

    echo "Running Thorin"
    $EXEC_DIR/runner.sh $EXEC_DIR/gmm/thorin.out $EXEC_DIR/gmm/data/$FOLDER $OUTPUT_DIR/thorin.csv

    echo "Running Enzyme"
    $EXEC_DIR/runner.sh $EXEC_DIR/gmm/enzyme.out $EXEC_DIR/gmm/data/$FOLDER $OUTPUT_DIR/enzyme.csv

    echo "Running PyTorch 2.0 CPU Single Thread"
    $EXEC_DIR/py_run.sh $EXEC_DIR/python_src/modules/PyTorch/PyTorchGMM2S.py $EXEC_DIR/gmm/data/$FOLDER $OUTPUT_DIR/pytorch.csv
done

