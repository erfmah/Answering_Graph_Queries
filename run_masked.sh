
#!/bin/bash



export LD_LIBRARY_PATH=/localhome/pnaddaf/anaconda3/envs/env/lib/
#"Cora" "ACM" "IMDB" "CiteSeer" "photos" "computers"
#for i in  "IMDB" "CiteSeer" "photos" "computers"
#do
#for j in "1"
#do
#for b in "single" "multi"
#do
#for a in "False" "True"
#do
#python -u main.py --dataSet "$i" --loss_type "$j"  --method "$b" --transductive "$a"
#done
#done
#done
#done

for i in  "IMDB" "Cora_dgl" "CiteSeer_dgl"
do
for j in "2"
do
for b in "single"
do
for a in "False"
do
for c in "Multi_GAT"
do
python -u main.py --dataSet "$i" --loss_type "$j"  --method "$b" --iterative "$a" --encoder_type "$c"
done
done
done
done
done
