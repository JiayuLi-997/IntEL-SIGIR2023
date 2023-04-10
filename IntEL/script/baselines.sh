cd ../src/
DATAPATH=../data
DATANAME=Tmall_toy
# single sort
python3 main.py --save_anno single_click --datapath $DATAPATH --dataset $DATANAME --runner_name BaseRunner --model_name SingleSort --batch_size 512 --num_workers 4 --gpu 2 --train 0 --topk 1,3,5,10 --regenerate 0 --max_session_len 100 --intent_note _multi --model_num 3 --choose_list pCTR
python3 main.py --save_anno single_fav --datapath $DATAPATH --dataset $DATANAME --runner_name BaseRunner --model_name SingleSort --batch_size 512 --num_workers 4 --gpu 2 --train 0 --topk 1,3,5,10 --regenerate 0 --max_session_len 100 --intent_note _multi --model_num 3 --choose_list pFVR
python3 main.py --save_anno single_buy --datapath $DATAPATH --dataset $DATANAME --runner_name BaseRunner --model_name SingleSort --batch_size 512 --num_workers 4 --gpu 2 --train 0 --topk 1,3,5,10 --regenerate 0 --max_session_len 100 --intent_note _multi --model_num 3 --choose_list pCVR

# RRA
python3 main.py --save_anno RRA --runner_name BaseRunner --datapath $DATAPATH --dataset $DATANAME --model_name RRA --batch_size 512 --train 0 --num_workers 4 --gpu 2 --topk 1,3,5,10 --regenerate 0 --max_session_len 100 --intent_note _multi --model_num 3

# Borda
python3 main.py --save_anno Borda --runner_name BaseRunner --datapath $DATAPATH --dataset $DATANAME --model_name Borda --batch_size 512 --train 0 --num_workers 4 --gpu 2 --topk 1,3,5,10 --regenerate 0 --max_session_len 100 --intent_note _multi --model_num 3

# Lambda-Rank
seed_list=( 1 2 3 4 5 )
for seed in ${seed_list[@]}
do
python3 main.py --random_seed ${seed} --save_anno lambdaRank_seed${seed} --datapath $DATAPATH --dataset $DATANAME --model_name LambdaRank --lr 2e-4 --runner_name LambdaRankRunner --batch_size 512 --num_workers 4 --gpu 2 --train 1 --topk 3,1,5,10 --main_metric NDCG@3 --max_session_len 100 --intent_note _multi --model_num 3 --hidden_size 128
done

# ERA
seed_list=( 1 2 3 4 5)
for seed in ${seed_list[@]}
do
python3 ERARunner.py --random_seed ${seed} --datapath $DATAPATH --dataset $DATANAME --intent_note _multi --max_session_len 100 --save_anno ERA_${i} --num_generations 10 --num_solutions 100 --num_parents_mating 5 --crossover_prob 0.65 --mutation_prob 0.25 --elitism 2 --gpu 3 --model_num 3 --topk 3,1,5,10 --main_metric NDCG@3
done

# aWELv
seed_list=( 1 2 3 4 5 )
for seed in ${seed_list[@]}
do
python3 main.py --save_anno aWELv_seed${seed} --random_seed ${seed} --runner_name BaseRunner --loss_name Listloss --datapath $DATAPATH --dataset $DATANAME --model_name aWELv --batch_size 512 --num_workers 4 --gpu 3 --topk 3,1,5,10 --regenerate 0 --test_epoch 5 --max_session_len 100 --intent_note _multi --model_num 3 --main_metric NDCG@3 --lr 2e-4 --l2 1e-4 --hidden_size 32 --cal_diversity 1 --diversity_alpha 1e-6
done

# aWELv+Int
seed_list=( 1 2 3 4 5 )
for seed in ${seed_list[@]}
do
python3 main.py --save_anno aWELv_Int_seed${seed} --random_seed ${seed} --runner_name BaseRunner --loss_name IntListloss --datapath $DATAPATH --dataset $DATANAME --model_name aWELv_Int --batch_size 512 --num_workers 4 --gpu 2 --topk 3,1,5,10 --regenerate 0 --test_epoch 5 --max_session_len 100 --intent_note _multi --model_num 3 --main_metric NDCG@3 --lr 2e-4 --l2 1e-4 --intent_weight 0.05 --context_emb_size 32 --intent_emb_size 32 --encoder GRU4Rec --i_emb_size 16 --im_emb_size 16 --user_emb_size 16 --cal_diversity 1 --diversity_alpha 1e-7
done

# aWELv+IntEL
seed_list=( 1 2 3 4 5 )
for seed in ${seed_list[@]}
do
python3 main.py --save_anno aWELv_IntEL_seed${seed} --random_seed ${seed} --runner_name BaseRunner --loss_name IntListloss --datapath $DATAPATH --dataset $DATANAME --model_name aWELv_IntEL --batch_size 512 --num_workers 4 --gpu 2 --topk 3,1,5,10 --regenerate 0 --test_epoch 5 --max_session_len 100 --intent_note _multi --model_num 3 --intent_weight 0.1 --kl_weight 0.5 --main_metric NDCG@3 --lr 1e-3 --l2 1e-4 --dropout 0.5 --context_emb_size 32 --intent_emb_size 32 --encoder GRU4Rec --i_emb_size 16 --im_emb_size 16 --u_emb_size 16 --s_emb_size 32 --cross_attn_qsize 64 --num_heads 2 --num_layers 2
done
