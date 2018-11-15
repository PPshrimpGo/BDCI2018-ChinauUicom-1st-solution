#!/bin/bash

stage=0

echo '**' `date +%H:%M:%S` 'start with stage=' $stage 
echo 'Any question contact QQ:674785731'

mkdir -p cv
mkdir -p sub
mkdir -p cache
mkdir -p data/a
mkdir -p data/b
cp input/train_old.csv data/a/train.csv
cp input/train.csv data/b/train_new.csv
cp input/test.csv data/b/test_new.csv


# gen magic feature
if [ $stage -le 0 ]; then
  cd feature && python3 get_most.py && cd ..
  echo '**' `date +%H:%M:%S` 'finished get most'
fi

# white
if [ $stage -le 1 ]; then
  cd feature && python3 white.py && cd ..
  echo '**' `date +%H:%M:%S` 'finished white'
fi
#model 1
if [ $stage -le 2 ]; then
  python3 model1.py
  echo '**' `date +%H:%M:%S` 'finished model1'
fi

#model 2
if [ $stage -le 3 ]; then
  python3 model2.py
  echo '**' `date +%H:%M:%S` 'finished model2'
fi

#model 3
if [ $stage -le 4 ]; then
  python3 model3_1.py
  python3 model3_4.py
  echo '**' `date +%H:%M:%S` 'finished model3'
fi

#model pred fee
if [ $stage -le 5 ]; then
  python3 model_pred_fee.py
  echo '**' `date +%H:%M:%S` 'finished model fee pred'
fi

# combibe 1 and 4
if [ $stage -le 6 ]; then
  python3 hebing_pred.py
  echo '**' `date +%H:%M:%S` 'finished combine'
fi

# piupiu
if [ $stage -le 7 ]; then
  python3 clean_a.py
  python3 clean_b.py
  python3 data_pred_a+2b.py
  python3 data_pred_a+b.py
  echo '**' `date +%H:%M:%S` 'finished clean'
fi

# stacking
if [ $stage -le 8 ]; then
  python3 baseline_v11.py
  echo '**' `date +%H:%M:%S` 'finished fianle model'
fi

# whilte
if [ $stage -le 9 ]; then
  python3 piupiu_white.py
  echo '**' `date +%H:%M:%S` 'finished white'
fi
echo 'all done! submit file is sub_final_white'
