if [ $# -ne 3 ]; then
    echo "Usage: $0 <python_module> <benchmark_folder> <output_file>"
    exit 1
fi

FILE=$3
MODULE=$1
DATA=$2
EXEC_DIR=/app/mount

source /opt/conda/bin/activate /app/env

TIMESTAMP=$(date +"%Y_%m_%d_%H_%M_%S")
TEMP_DIR="$EXEC_DIR/logs/$TIMESTAMP"
if [ -d "$TEMP_DIR" ]; then
    rm -rf $TEMP_DIR
fi
mkdir -p $TEMP_DIR
python runner.py -m $MODULE -d $DATA -o $TEMP_DIR
python collect_gmm_data.py -f $TEMP_DIR -o $FILE
