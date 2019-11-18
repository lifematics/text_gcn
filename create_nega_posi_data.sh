#! /bin/bash
function usage() {
cat <<_EOT_
Usage:
  bash create_nega_posi_data.sh [-d dataset name] [-p positives] [-n negatives] [-t prediction targets] [-r test size rate] [-l cut off file length]

Options:
  -d --dataset   dataset name.  default: np
  -p --positive   positive data directory.  default: data/<dataset name>/positives
  -n --negative   negative data directory.  default: data/<dataset name>/negatives
  -t --target   prediction target data directory.  default: data/<dataset name>/targets
  -r --test-size   percentage of test data size.(not sample number.)  default: 10
  -l --min-length minimum string length of the file to be truncated.  default: 20
_EOT_
return 0
}

DIR="np"
POS="data/$DIR/positives"
NEG="data/$DIR/negatives"
TGT="data/$DIR/targets"
TEST_RATE=10
PROC=true
MIN_LEN=20
while getopts "dnpr:h-:" OPT
do
    case $OPT in
        -)
            case "${OPTARG}" in
                positive)  POS=$OPTARG
                    ;;
                negative)  NEG=$OPTARG
                    ;;
                target)  TGT=$OPTARG
                    ;;
                dataset)  DIR=$OPTARG
                    ;;
                test-size)  TEST_RATE=$OPTARG
                    ;;
                min-length)  MIN_LEN=$OPTARG
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
        t)  TGT=$OPTARG
            ;;
        d)  DIR=$OPTARG
            ;;
        r)  TEST_RATE=$OPTARG
            ;;
        l)  MIN_LEN=$OPTARG
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

index_table="data/$DIR.txt"
corpus_file="data/corpus/$DIR.txt"
target_table="data/${DIR}_targets.txt"

cp /dev/null $index_table
cp /dev/null $corpus_file
cp /dev/null $target_table

pos_files="$POS/*"
neg_files="$NEG/*"
target_files="$TGT/*"

i=0
if $PROC; then
  for filepath in $pos_files; do
    str="$(cat $filepath)"
    if [ ${#str} -lt $MIN_LEN ]; then
      echo $filepath has too few words.
      continue
    fi
    if [ $((RANDOM%+100)) -lt $TEST_RATE ]; then
      echo -e "$i\ttest\tpositive" >> $index_table
    else
      echo -e "$i\ttrain\tpositive" >> $index_table
    fi
    echo -e "$str" >> $corpus_file
    i=$((i+1))
  done

  for filepath in $neg_files; do
    if [ ${#str} -lt $MIN_LEN ]; then
      echo $filepath has too few words.
      continue
    fi
    str="$(cat $filepath)"
    if [ $((RANDOM%+100)) -lt $TEST_RATE ]; then
      echo -e "$i\ttest\tnegative" >> $index_table
    else
      echo -e "$i\ttrain\tnegative" >> $index_table
    fi
    echo -e "$str" >> $corpus_file
    i=$((i+1))
  done
  if [ -e $TGT ]; then
    for filepath in $target_files; do
      if [ ${#str} -lt $MIN_LEN ]; then
        echo $filepath has too few words.
        continue
      fi
      str="$(cat $filepath)"
      echo -e "$i\ttarget\tnegative" >> $index_table
      echo -e "$str" >> $corpus_file
      echo -e "$i\t$filepath" >> $target_table
      i=$((i+1))
    done
  fi
fi
unset DIR POS NEG TGT TEST_RATE PROC OPT OPTARG OPTIND MIN_LEN