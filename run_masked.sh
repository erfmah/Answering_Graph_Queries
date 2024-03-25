
#!/bin/bash



export LD_LIBRARY_PATH=/localhome/pnaddaf/anaconda3/envs/env/lib/
#"Cora" "ACM" "IMDB" "CiteSeer" "photos" "computers"
for i in "computers"
do
for j in '1'
do
for a in "Multi_GIN"
do
for b in "multi"
do
python -u main.py --dataSet "$i" --loss_type "$j" --encoder_type "$a" --method "$b"
done
done
done
done
