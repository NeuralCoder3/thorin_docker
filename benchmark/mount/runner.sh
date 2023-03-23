# assert three arguments are given
if [ $# -ne 3 ]; then
    echo "Usage: $0 <executable> <benchmark_folder> <output>"
    exit 1
fi

FILE=$3
EXECUTABLE=$1
FILES=$2/*.txt

echo -e "file\ttime" > $FILE
SORTED=$(ls -v $FILES)

for f in $SORTED ; do
    echo "Processing $f file..."
    # TIME=$(taskset -c 1 $EXECUTABLE $f | head -n 1)
    TIME=$($EXECUTABLE $f | head -n 1)
    # error in execution or no time
    if [ $? -ne 0 ] || [ -z "$TIME" ]; then
        echo "Error in execution, command:"
        echo -e "\t$EXECUTABLE $f"
        continue
    fi
    name=$(basename $f .txt)
    echo -e "$name\t$TIME" >> $FILE
    echo "Time: $TIME"
done
