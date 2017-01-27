export DSUP_EVENT_DIR=/home/konovalo/dsup_event
export OUTPUT_DIR=../experiments_gs5

for attribute in `cat $1`
do
    echo gridsearch on $attribute
    mkdir -p $OUTPUT_DIR
    nohup python $DSUP_EVENT_DIR/python/gridsearch.py $attribute $OUTPUT_DIR 1>/dev/null &
done
