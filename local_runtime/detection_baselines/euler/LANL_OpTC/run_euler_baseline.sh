python run.py --dataset LANL --hidden 32 -z 16 --rnn GRU -d 0.5 --lr 0.005 --fpweight 0.6 --patience 5 -t 5
sleep 30
mv model_save.pkl model_save_lanl.pkl
python run.py --dataset OPTC --hidden 32 -z 16 --rnn LSTM -d 2.5 --lr 0.005 --fpweight 0.6 --patience 5 -t 5
sleep 30
mv model_save.pkl model_save_optc.pkl
