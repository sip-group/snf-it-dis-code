for seed in 0 1 2 3 4
do
  for run in 1 2 3 4 5 6
  do
    for n in {1..100}
    do
      for phone in samsung iphone
      do
        echo "train on $phone with nb_samples : $n , run : $run seed : $seed"
        python3 measure_codebook.py -n $n -d $phone -r $run -c 8 -s $seed
      done
    done
  done
done


# shutdown -h now
