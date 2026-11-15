python3 run.py --dataset oilrig  --hidden 32 -z 16 --rnn GRU -d 2.5 --lr 0.005 --fpweight 0.48 --patience 10 -t 5
sleep 30
mv model_save.pkl model_save_oilrig.pkl
python3 run.py --dataset wizard  --hidden 32 -z 16 --rnn GRU -d 2.5 --lr 0.005 --fpweight 0.47 --patience 10 -t 5
sleep 30
mv model_save.pkl model_save_wizard.pkl
python3 run.py --dataset sandworm  --hidden 32 -z 16 --rnn GRU -d 2.5 --lr 0.005 --fpweight 0.42 --patience 10 -t 5
sleep 30
mv model_save.pkl model_save_sandworm.pkl
