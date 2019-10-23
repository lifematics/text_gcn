#! /bin/bash
function usage() {
cat <<_EOT_
Usage:
  bash create_nega_posi_data.sh [-d dataset name] [-p positives] [-n negatives] [-r test size rate]

Options:
  -d --dataset   dataset name.  default: np
  -p --positive   positive data directory.  default: data/<dataset name>/positives
  -n --negative   negative data directory.  default: data/<dataset name>/negatives
  -r --test-size   percentage of test data size.(not sample number.)  default: 10
_EOT_
return 0
}

DIR="np"
POS="data/$DIR/positives"
NEG="data/$DIR/negatives"
TEST_RATE=10
PROC=true
while getopts "dnpr:h-:" OPT
do
    case $OPT in
        -)
            case "${OPTARG}" in
                positive)  POS=$OPTARG
                    ;;
                negative)  NEG=$OPTARG
                    ;;
                dataset)  DIR=$OPTARG
                    ;;
                test-size)  TEST_RATE=$OPTARG
                    ;;
                help)
                    usage
                    PROC=false
                    ;;
            esac
            ;;
        p)  POS=$OPTARG
            ;;
        n)  NEG=$OPTARG
            ;;
        d)  DIR=$OPTARG
            ;;
        r)  TEST_RATE=$OPTARG
            ;;
        h)  usage
            PROC=false
            ;;
        \?) usage
            PROC=false
            ;;
    esac
done
shift $((OPTIND - 1))

cp /dev/null "data/corpus/$DIR.txt"
cp /dev/null "data/$DIR.txt"

pos_files="$POS/*"
neg_files="$NEG/*"

i=0
if $PROC; then
  for filepath in $pos_files; do
    str="$(cat $filepath)"
    if [ ${#str} -lt 5 ]; then
      echo $filepath
    fi
    if [ $((RANDOM%+100)) -lt $TEST_RATE ]; then
      echo -e "$i\ttest\tpositive" >> "data/$DIR.txt"
    else
      echo -e "$i\ttrain\tpositive" >> "data/$DIR.txt"
    fi
    echo -e "$str" >> "data/corpus/$DIR.txt"
    i=$((i+1))
  done

  for filepath in $neg_files; do
    if [ ${#str} -lt 5 ]; then
      echo $filepath
    fi
    str="$(cat $filepath)"
    if [ $((RANDOM%+100)) -lt $TEST_RATE ]; then
      echo -e "$i\ttest\tnegative" >> "data/$DIR.txt"
    else
      echo -e "$i\ttrain\tnegative" >> "data/$DIR.txt"
    fi
    echo -e "$str" >> "data/corpus/$DIR.txt"
    i=$((i+1))
  done
fi
unset DIR POS NEG TEST_RATE PROC OPT OPTARG OPTIND