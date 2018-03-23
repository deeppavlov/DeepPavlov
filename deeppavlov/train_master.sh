# #config
# model_set[0]='bah_large_bs16.json bah_ligth_bs16.json gen_large_bs16.json gen_ligth_bs16.json gen_cs_large_bs16.json bah_cs_large_bs16.json'
# model_set[1]='bah_large_bs1.json bah_ligth_bs1.json gen_large_bs1.json gen_ligth_bs1.json gen_cs_large_bs1.json bah_cs_large_bs1.json'
# model_set[2]='bah_large_projected_bs16.json bah_ligth_projected_bs16.json gen_large_projected_bs16.json gen_ligth_projected_bs16.json gen_cs_large_projected_bs16.json bah_cs_large_projected_bs16.json'
# model_set[3]='bah_large_projected_bs1.json bah_ligth_projected_bs1.json gen_large_projected_bs1.json gen_ligth_projected_bs1.json gen_cs_large_projected_bs1.json bah_cs_large_projected_bs1.json'
#config
model_set[0]='gen_cs_large_bs16.json bah_cs_large_bs16.json'
model_set[1]='gen_cs_large_bs1.json bah_cs_large_bs1.json'
model_set[2]='gen_cs_large_projected_bs16.json bah_cs_large_projected_bs16.json'
model_set[3]='gen_cs_large_projected_bs1.json bah_cs_large_projected_bs1.json'
gpus[0]=0
gpus[1]=1
gpus[2]=2
gpus[3]=6
#start
# setenvdp-go-gpu
CUDA_VISIBLE_DEVICES=${gpus[$1]}
for i in  ${model_set[$1]};
do
echo "Used model is $i "'#START_TOKEN ';;
python deep.py train configs/go_bot/dev/$i
echo "Used model is $i "'#FIND_TOKEN ';
done
