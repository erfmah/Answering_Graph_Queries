
#!/bin/bash



export LD_LIBRARY_PATH=/localhome/pnaddaf/anaconda3/envs/env/lib/
#"Cora" "ACM" "IMDB" "CiteSeer" "photos" "computers"
for i in "Cora" "IMDB" "CiteSeer" "photos" "computers"
do
for j in "۱"
do
for b in "single"
do
for a in "False"
do
python -u main.py --dataSet "$i" --loss_type "$j"  --method "$b" --transductive "$a"
done
done
done
done