
if [ $# -ge 1 ]
then
execute_type=$1
else
execute_type=list
fi

if [ $# -ge 2 ]
then
NUM_Process=$2
else
NUM_Process=100
fi

for i in $( seq 0 ${NUM_Process} )
# for i in $( seq 0 1 )
do
    python3 utils_hdfs.py ${i} ${NUM_Process} ${execute_type}
done 
