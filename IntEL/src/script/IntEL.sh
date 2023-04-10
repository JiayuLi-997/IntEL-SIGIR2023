cd ../
DATAPATH=../../../../data_new_0109
DATANAME=All_deepfm_0109_300cate_len30
seed_list=( 1 2 3 4 5 )

# IntEL-MSE 
for seed in ${seed_list[@]}
do
python3 main.py --save_anno IntEL_MSE_seed${seed} --random_seed ${seed} --runner_name BaseRunner --loss_name IntMSEloss --datapath $DATAPATH --dataset $DATANAME --model_name IntEL --batch_size 512 --num_workers 4 --gpu 2 --topk 3,1,5,10 --regenerate 0 --test_epoch 3 --max_session_len 100 --intent_note _multi --model_num 3 --intent_weight 0.003 --kl_weight 0.5 --main_metric NDCG@3 --encoder BERT4Rec --lr 1e-3 --l2 1e-6 --dropout 0.5 --cal_diversity 1 --diversity_alpha 1e-5
done

# # IntEL-BPR
# for seed in ${seed_list[@]}
# do
# python3 main.py --save_anno IntEL_BPR_seed${seed} --random_seed ${seed} --runner_name BaseRunner --loss_name IntBPRloss --datapath $DATAPATH --dataset $DATANAME --model_name IntEL --batch_size 512 --num_workers 4 --gpu 2 --topk 3,1,5,10 --regenerate 0 --test_epoch 3 --max_session_len 100 --intent_note _multi --model_num 3 --intent_weight 0.01 --kl_weight 0.5 --main_metric NDCG@3 --lr 1e-4 --l2 1e-4 --dropout 0 --context_emb_size 64 --intent_emb_size 32 --encoder GRU4Rec --i_emb_size 16 --im_emb_size 16 --u_emb_size 32 --s_emb_size 32 --cal_diversity 1 --diversity_alpha 1e-3 --cross_attn_qsize 32 --num_heads 2 --num_layers 2
# done

# # IntEL-PL
# for seed in ${seed_list[@]}
# do
# python3 main.py --save_anno IntEL_List_seed${seed} --random_seed ${seed} --runner_name BaseRunner --loss_name IntListloss --datapath $DATAPATH --dataset $DATANAME --model_name IntEL --batch_size 512 --num_workers 4 --gpu 2 --topk 3,1,5,10 --regenerate 0 --test_epoch 5 --max_session_len 100 --intent_note _multi --model_num 3 --intent_weight 0.1 --kl_weight 0.5 --main_metric NDCG@3 --lr 2e-3 --l2 1e-4 --dropout 0 --decay_lr 0 --context_emb_size 32 --intent_emb_size 32 --encoder GRU4Rec --i_emb_size 16 --im_emb_size 16 --u_emb_size 32 --s_emb_size 32 --cross_attn_qsize 64 --num_heads 2 --num_layers 2 --cal_diversity 1 --diversity_alpha 1e-4
# done