for train_phone in iphone samsung
do
  for test_phone in iphone samsung
  do
    for run in 1 2 3 4 5 6
    do
      for seed in 1 4 5
      do
        echo "train on $train_phone with nb_samples : 50 , run : $run seed : $seed"
        python3 measure_metrics.py -n 50 -t $train_phone -d $test_phone -r $run -c 8 -s $seed -v
      done
    done
  done
done


shutdown -h now
