DATASET_DIR = './dataset/'
TRAIN_FILE = 'train'
VAL_FILE = 'val'
TEST_FILE = 'test'

USER_FILE = 'user'
ITEM_FILE = 'item'

VAL_IIDS_FILE = 'val_iids'  # DATASET_DIR/{dataset}/VAL_IIDS_FILE.{suffix}为验证集评测商品候选列表文件
TEST_IIDS_FILE = 'test_iids'  # DATASET_DIR/{dataset}/TEST_IIDS_FILE.{suffix}为测试集评测商品候选列表文件

LABEL = 'label'  # 标签

UID = 'u_id_c'  # 用户ID
IID = 'i_id_c'  # 商品ID
TIME = 'c_time_i'  # 时间
EVAL_IIDS = 'c_eval_iids_s'  # 候选集合商品ID列表
EVAL_LABELS = 'c_eval_labels_s'  # 候选集合标签列表，该列应与EVAL_IIDS长度相同且一一对应

import pandas as pd
import numpy as np
import os
import rich.progress

def group_user_history(uids, iids) -> dict:
    user_dict = {}
    for uid, iid in zip(uids, iids):
        if uid not in user_dict:
            user_dict[uid] = []
        user_dict[uid].append(iid)
    return user_dict

def group_user_history_df(df, label_filter=lambda x: x > 0) -> dict:
    dfs = [df] if type(df) is pd.DataFrame else df
    dfs = [df[df[LABEL].apply(label_filter)] if LABEL in df else df for df in dfs]
    df = pd.concat([df[[UID, IID]] for df in dfs], ignore_index=True)
    return group_user_history(uids=df[UID].values, iids=df[IID].values)

def create_pgbar(iters, total=None, desc='', leave=False):
    args = {'refresh_per_second': 1, 'total': total, 'description': desc, 'transient': not leave}
    return rich.progress.track(iters, **args)

def sample_iids(sample_n, uids, item_num, exclude_iids=None, replace=False, item_p=None):
    uids_len = len(uids)
    uids = create_pgbar(uids, desc='sample_iids')

    global_exclude = None if type(exclude_iids) is dict \
        else set([]) if exclude_iids is None else set(exclude_iids)  # 判断exclude_iids是不是字典，不是的话是全局去除商品ID
    if global_exclude is None:  # exclude_iids是对每个用户去除的采样ID
        exclude_iids = {k: set(exclude_iids[k]) for k in exclude_iids}  # 用set判断更快

    # 先批量采样大量ID作为缓存。使用时从缓存中读再去除不合适的，这样其实更快
    if item_p is None:
        iid_buffer = np.random.randint(low=0, high=item_num, size=sample_n * uids_len)
    else:
        assert len(item_p) == item_num
        iid_buffer = np.random.choice(item_num, size=sample_n * uids_len, replace=True, p=item_p)
    buffer_idx = 0  # 记录当前缓存使用的位置
    result = []
    for uid in uids:  # 对每个样本所对应的用户ID
        exclude = global_exclude if global_exclude is not None \
            else exclude_iids[uid] if uid in exclude_iids else set([])
        if not replace and item_num - len(exclude) < sample_n:  # 如果剩下可采样的ID不够sample_n个
            uid_result = [i for i in range(item_num) if i not in exclude]
            while len(uid_result) != sample_n:
                uid_result.append(0)  # 补0
            result.append(uid_result)
            continue
        uid_result = []
        tmp_set = set([])
        while len(uid_result) < sample_n:  # 对该样本还没采样够
            if len(iid_buffer) <= buffer_idx:  # 如果缓存的采样商品ID用完了，重新采样一个缓存
                if item_p is None:
                    iid_buffer = np.random.randint(low=0, high=item_num, size=sample_n * uids_len)
                else:
                    iid_buffer = np.random.choice(item_num, size=sample_n * uids_len, replace=True, p=item_p)
                buffer_idx = 0
            iid = iid_buffer[buffer_idx]
            buffer_idx += 1
            if iid not in exclude and (replace or iid not in tmp_set):  # 查看缓存中的ID是否合适
                uid_result.append(iid)
                tmp_set.add(iid)
        result.append(uid_result)
    return np.array(result)

def random_sample_eval_iids(dataset_name, sample_n=1000, include_neg=False):
    dir_name = os.path.join(DATASET_DIR, dataset_name)
    train_df = pd.read_csv(os.path.join(dir_name, TRAIN_FILE + '.csv'), sep='\t')
    val_df = pd.read_csv(os.path.join(dir_name, VAL_FILE + '.csv'), sep='\t')
    test_df = pd.read_csv(os.path.join(dir_name, TEST_FILE + '.csv'), sep='\t')
    item_df = pd.read_csv(os.path.join(dir_name, ITEM_FILE + '.csv'), sep='\t')
    item_num = len(item_df[IID])

    def eval_iids(in_df, user_his):
        uls = list(zip(in_df[UID].values, in_df[LABEL].values))
        sample_neg = sample_iids(sample_n=sample_n, uids=[uid for uid, label in uls if label > 0],
                                 item_num=item_num, exclude_iids=user_his)  # 对每个正向交互从全局商品中随机采样一些不包括已知用户正向交互
        sample_neg = [[str(i) for i in li] for li in sample_neg]
        eval_c = [sample_neg.pop(0) if label > 0 else [] for uid, label in uls]  # 负向交互不需要采样负例候选集
        assert len(sample_neg) == 0
        if include_neg:  # 如果要包括测试集/验证集中已知负例
            user_neg = group_user_history_df([in_df], label_filter=lambda x: x <= 0)
            for idx, (uid, label) in enumerate(uls):
                if label > 0 and uid in user_neg:
                    for i in range(min(len(eval_c[idx]), len(user_neg[uid]))):
                        eval_c[idx][i] = str(user_neg[uid][i])  # 替换采样候选集中前i个为已知负例
        eval_c = [','.join(i) for i in eval_c]  # 逗号连接
        eval_c = pd.DataFrame(data={EVAL_IIDS: eval_c})
        return eval_c

    # test
    tvt_user_his = group_user_history_df([train_df, val_df, test_df])
    test_iids = eval_iids(test_df, tvt_user_his)
    test_iids.to_csv(os.path.join(dir_name, TEST_IIDS_FILE + '.csv'), sep='\t', index=False)
    # val
    tv_user_his = group_user_history_df([train_df, val_df])  # 验证集不能看到测试集
    val_iids = eval_iids(val_df, tv_user_his)
    val_iids.to_csv(os.path.join(dir_name, VAL_IIDS_FILE + '.csv'), sep='\t', index=False)
    return

for i in range(1, 5):
    for label_type in ['click', 'buy', 'favorite']:
        dataset_name = f'Tmall_AugSep_window{i}/{label_type}'
        random_sample_eval_iids(dataset_name, sample_n=1000, include_neg=True)
