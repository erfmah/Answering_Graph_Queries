
#!/bin/bash



export LD_LIBRARY_PATH=/localhome/pnaddaf/anaconda3/envs/env/lib/
#"Cora" "ACM" "IMDB" "CiteSeer" "photos" "computers"
for i in "Cora" "ACM"
do
for j in "0" "3"
do
for b in "single" "multi"
do
for a in "False" "True"
do
python -u main.py --dataSet "$i" --loss_type "$j"  --method "$b" --transductive "$a"
done
done
done
done