#!/bin/bash
DIR="np"
POS="data/$DIR/positives"
NEG="data/$DIR/negatives"
TEST_RATE=10
PROC=true

cp /dev/null "data/corpus/$DIR.txt"
cp /dev/null "data/$DIR.txt"

pos_files="$POS/*"
neg_files="$NEG/*"

i=0
if $PROC; then
  for filepath in $pos_files; do
    str="$(cat $filepath)"
    if [ $((RANDOM%+100)) -lt $TEST_RATE ]; then
      echo -e "$i\ttest\tpositive" >> "data/$DIR.txt"
    else
      echo -e "$i\ttrain\tpositive" >> "data/$DIR.txt"
    fi
    echo -e "$str" >> "data/corpus/$DIR.txt"
    i=$((i+1))
  done

  for filepath in $neg_files; do
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
return 0