import os, sys
import os.path as osp
import shutil

from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.tracking.streamline import select_random_set_of_streamlines
from dipy.tracking.streamlinespeed import set_number_of_points
import nibabel as nib
import torch
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import tqdm
from tqdm import trange
import pickle
import argparse


HCP105_names = [
    'AF_left', 'AF_right', 'CA', 'CG_left', 'CG_right', 'CST_left', 'CST_right',
    'MLF_left', 'MLF_right', 'FPT_left', 'FPT_right', 'FX_left', 'FX_right',
    'ICP_left', 'ICP_right', 'IFO_left', 'IFO_right', 'ILF_left', 'ILF_right',
    'MCP', 'POPT_left', 'POPT_right', 'SCP_left', 'SCP_right', 'SLF_I_left',
    'SLF_I_right', 'SLF_II_left', 'SLF_II_right', 'SLF_III_left',
    'SLF_III_right', 'STR_left', 'STR_right', 'UF_left', 'UF_right',
    'T_PREM_left', 'T_PREM_right', 'T_PREC_left', 'T_PREC_right',
    'ST_PREM_left', 'ST_PREM_right', 'ST_PREC_left', 'ST_PREC_right',
    'ST_OCC_left', 'ST_OCC_right', 'ATR_left', 'ATR_right', 'T_POSTC_left',
    'T_POSTC_right', 'OR_left', 'OR_right', 'ST_POSTC_left', 'ST_POSTC_right',
    'ST_FO_left', 'ST_FO_right', 'T_PREF_left', 'T_PREF_right', 'T_PAR_left',
    'T_PAR_right', 'T_OCC_left', 'T_OCC_right', 'ST_PAR_left', 'ST_PAR_right',
    'ST_PREF_left', 'ST_PREF_right',
    'CC', 'CC_1', 'CC_2', 'CC_3', 'CC_4', 'CC_5', 'CC_6', 'CC_7',
    'T_PREF_left+ATR_left', 'T_PREF_right+ATR_right', 'T_PAR_left+T_POSTC_left', 
    'T_PAR_right+T_POSTC_right', 'T_OCC_left+OR_left', 'T_OCC_right+OR_right',
    'ST_PAR_left+ST_POSTC_left', 'ST_PAR_right+ST_POSTC_right',
    'ST_PREF_left+ST_FO_left', 'ST_PREF_right+ST_FO_right'
]

HCP105_name2id = {key:value for value, key in enumerate(HCP105_names)}
HCP105_id2name = {value:key for key, value in HCP105_name2id.items()}

include_relation = {
    'T_PREF_left': ['ATR_left'], 'T_PREF_right': ['ATR_right'],
    'T_PAR_left': ['T_POSTC_left'], 'T_PAR_right': ['T_POSTC_right'],
    'T_OCC_left': ['OR_left'], 'T_OCC_right': ['OR_right'],
    'ST_PAR_left': ['ST_POSTC_left'], 'ST_PAR_right': ['ST_POSTC_right'],
    'ST_PREF_left': ['ST_FO_left'], 'ST_PREF_right': ['ST_FO_right']
}

npoints = 20


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type = str, default='./data/HCP105', help = 'dataset path')
    parser.add_argument('--dp_rate', type = int, default=1000, help = 'downsample rate')
    parser.add_argument('--n_thread', type = int, default=1, help = 'number of thread')

    parser.add_argument(
        '--update_trk_files',
        action='store_true',
        default=False,
        help = 'find overlap fibers and create a new pkl file for them')
    parser.add_argument(
        '--check_duplicate_num',
        action='store_true',
        default=False,
        help = 'computer the number of duplicate fibers between two trk files')
    parser.add_argument(
        '--data_clean',
        action='store_true',
        default=False,
        help = 'read the trk files and save as dict')
    parser.add_argument(
        '--check_duplicate_name',
        action='store_true',
        default=False,
        help = 'find fibers with multi-label')
    parser.add_argument(
        '--add_cr_label',
        action='store_true',
        default=False,
        help = 'generate fa label')
    parser.add_argument(
        '--clean_data_dir',
        action='store_true',
        default=False,
        help = 'delete some unused files')
    parser.add_argument(
        '--data_preprocess',
        action='store_true',
        default=False,
        help = 'make all fibers with the same direction')
    parser.add_argument(
        '--process_csv',
        action='store_true',
        default=False,
        help = 'process HCP csv file')
    parser.add_argument(
        '--compute_fiber_distance',
        action='store_true',
        default=False,
        help = 'computer distance between two kinds of fiber')
    
    args = parser.parse_args()

    return args


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    try:
        farthest = np.random.randint(0, N)
    except:
        import ipdb; ipdb.set_trace()
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    return centroids.astype(np.int32)


def random_sample(streamlines, sampling_num):
    streamlines_num = len(streamlines)
    index_list = [a for a in range(streamlines_num)]
    sampling_list = np.random.choice(index_list, size=sampling_num, replace=False)
    return sampling_list


def fiber_distance_cal_Efficient(set1, set2, num_points=20):
    set1 = torch.from_numpy(set1)
    set2 = torch.from_numpy(set2)
    set1 = set1.reshape(set1.shape[0], -1)
    set2 = set2.reshape(set2.shape[0], -1)
    set1_squ = (set1 ** 2).sum(1).view(-1, 1)
    set2_t = torch.transpose(set2, 0, 1)
    set2_squ = (set2 ** 2).sum(1).view(1, -1)
    dist = set1_squ + set2_squ - 2.0 * torch.mm(set1, set2_t)
    dist = torch.sqrt(torch.clamp(dist, 0.0, np.inf))

    mean_dist = torch.div(dist, num_points)
    
    return mean_dist


def fps_sample(streamlines, sample_streamlines, sampling_num):
    dist_1 = fiber_distance_cal_Efficient(sample_streamlines, sample_streamlines)
    reverse_streamlines = sample_streamlines[:, ::-1, :]
    reverse_streamlines = reverse_streamlines.copy()
    dist_2 = fiber_distance_cal_Efficient(sample_streamlines, reverse_streamlines)

    for i in range(len(sample_streamlines)):
        if dist_1[0][i] > dist_2[0][i]:
            sample_streamlines[i] = sample_streamlines[i, ::-1, :]
            streamlines[i] = streamlines[i][::-1, :]

    if sampling_num >= len(streamlines):
        sampling_list = np.arange(0, len(streamlines), 1)
        return sampling_list.astype(np.int32), streamlines, sample_streamlines

    dist = torch.min(dist_1, dist_2)

    N = sample_streamlines.shape[0]
    sampling_list = np.zeros((sampling_num,))
    farthest = np.random.randint(0, N)
    for i in range(sampling_num):
        sampling_list[i] = farthest
        distance = dist[farthest]
        rank = torch.sort(distance, descending=True)[1]

        j = 0
        while rank[j].item() in sampling_list:
            j += 1
        
        farthest = rank[j]

    del dist_1, dist_2, dist

    return sampling_list.astype(np.int32), streamlines, sample_streamlines


def save_fiber_distribution(pid, total_fibers, point_num_dict):
    print('Subject {} has {} fibers'.format(pid, total_fibers))
    with open('point_distribution.txt', 'a') as f:
        f.write('-----{}-----\n'.format(pid))
        print('-----{}-----'.format(pid))
        for key in point_num_dict.keys():
            point_sum = sum(point_num_dict[key])
            fiber_num = len(point_num_dict[key])
            if fiber_num != 0:
                f.write('{}: mean_point_num={}, std={}\n'.format(key, point_sum/fiber_num, np.var(point_num_dict[key])))
                print('{}: mean_point_num={}, std={}'.format(key, point_sum/fiber_num, np.var(point_num_dict[key])))


def run_process(single_process_func, person_dir_list, cpu_worker_num=16, downsample_rate=1000):
    from multiprocessing import Process

    item_count = len(person_dir_list)
    interval = item_count / cpu_worker_num
    process_pool = []

    for i in range(cpu_worker_num):
        start_index = int(i*interval)
        end_index = int((i+1)*interval)
        if i == cpu_worker_num - 1:
            process_pool.append(Process(target=single_process_func, args=(person_dir_list[start_index:], downsample_rate,)))
            print('Process {}: {} - end'.format(i, start_index))
        else:
            process_pool.append(Process(target=single_process_func, args=(person_dir_list[start_index:end_index], downsample_rate,)))
            print('Process {}: {} - {}'.format(i, start_index, end_index))
    [p.start() for p in process_pool]
    [p.join() for p in process_pool]
    [p.close() for p in process_pool]


def process_one_class(trk_label, streamlines):
    fiber_data_list = []

    streamlines_num = len(streamlines)
    if streamlines_num == 0:
        import ipdb; ipdb.set_trace()
    
    sampling_num = streamlines_num // downsample_rate
    if sampling_num < 100:
        sampling_num = min(streamlines_num, 100)

    sample_streamlines = []
    for i in range(len(streamlines)):
        np.random.seed(0)
        sample_streamlines.append(set_number_of_points(streamlines[i], npoints))

    sample_streamlines = np.array(sample_streamlines)
    # sampling_list, streamlines, sample_streamlines = fps_sample(streamlines, sample_streamlines, sampling_num)
    sampling_list = random_sample(streamlines, sampling_num)

    for index in sampling_list:
        original_fiber = np.array(streamlines[index])
        fiber = np.array(sample_streamlines[index])
        fiber_data_list.append({'label': trk_label, 'original_data': original_fiber, 'data': fiber})

    return fiber_data_list


def data_clean(person_dir_list, downsample_rate):
    tbar = tqdm.tqdm(total=len(person_dir_list))
    ignore_list = []

    for key in include_relation:
        ignore_list.append(key)
        for sub_class in include_relation[key]:
            ignore_list.append(sub_class)

    fx_list = [
        '912447', '887373', '680957', '922854', '984472'
    ]

    for person_dir in person_dir_list:
        save_path = osp.join(dir_in, person_dir)

        if osp.isdir(save_path):
            pid = person_dir
            fiber_data_list = []

            for key in include_relation:
                pkl_path = osp.join(dir_in, person_dir, '{}.pkl'.format(key))
                trk_label = key
                with open(pkl_path, 'rb') as f:
                    streamlines = pickle.load(f)
                
                if len(streamlines) != 0:
                    tmp_fiber_list = process_one_class(trk_label, streamlines)
                    fiber_data_list.extend(tmp_fiber_list)

                for sub_class in include_relation[key]:
                    pkl_path = osp.join(dir_in, person_dir, '{}+{}.pkl'.format(key, sub_class))
                    trk_label = '{}+{}'.format(key, sub_class)
                    with open(pkl_path, 'rb') as f:
                        streamlines = pickle.load(f)
                    
                    if len(streamlines) != 0:
                        tmp_fiber_list = process_one_class(trk_label, streamlines)
                        fiber_data_list.extend(tmp_fiber_list)

                    pkl_path = osp.join(dir_in, person_dir, '{}.pkl'.format(sub_class))
                    trk_label = sub_class
                    if not os.path.exists(pkl_path):
                        continue
                    with open(pkl_path, 'rb') as f:
                        streamlines = pickle.load(f)
                    
                    if len(streamlines) != 0:
                        tmp_fiber_list = process_one_class(trk_label, streamlines)
                        fiber_data_list.extend(tmp_fiber_list)

            for tract_file in os.listdir(osp.join(dir_in, person_dir, 'tracts')):
                if 'trk' not in tract_file:
                    continue

                trk_path = osp.join(dir_in, person_dir, 'tracts', tract_file)
                trk_label = tract_file.split('.')[0]

                if trk_label in ignore_list or trk_label not in HCP105_names:
                    continue
                
                if pid in fx_list and trk_label in ['FX_left', 'FX_right']:
                    continue

                sl_file = nib.streamlines.load(trk_path)
                streamlines = sl_file.streamlines
                
                if len(streamlines) != 0:
                    tmp_fiber_list = process_one_class(trk_label, streamlines)
                    fiber_data_list.extend(tmp_fiber_list)
                else:
                    print('subject {}, {} = 0'.format(pid, trk_label))
            
            save_point_set_as_vtk(save_path, 'whole_brain_{}.vtk'.format(downsample_rate), fiber_data_list)

            pkl_name = osp.join(save_path, '82_clean_data_{}.pkl'.format(downsample_rate))
            with open(pkl_name, 'wb') as f:
                pickle.dump(fiber_data_list, f, pickle.HIGHEST_PROTOCOL)
            
        tbar.update(1)

    tbar.close()


def data_preprocess(person_dir_list, downsample_rate):
    tbar = tqdm.tqdm(total=len(person_dir_list))
    subject_based_data_dict = {}

    fx_list = [
        '912447', '887373', '680957', '922854', '984472'
    ]

    for person_dir in person_dir_list:
        fiber_num, points_num = 0, 0
        points_sum = np.zeros(3)
        save_path = osp.join(dir_in, person_dir)

        if osp.isdir(save_path):
            pid = person_dir
            subject_based_data_dict = {
                'points_mean': None,
                'fiber_points_list': [],
                'pid': pid
            }

            fiber_data_list = []
            fiber_id_2_original_fiber = {}
            pkl_name = os.path.join(save_path, '82_clean_data_{}.pkl'.format(downsample_rate))

            with open(pkl_name, 'rb') as f:
                subject_data = pickle.load(f)

            for item in subject_data:
                if 'CC+' in item['label']:
                    item['label'] = item['label'][3:]
                
                point_set = item['data']
                original_point_set = item['original_data']
                trk_label = item['label']

                if trk_label not in HCP105_names:
                    continue
                if pid in fx_list and (trk_label == 'FX_left' or trk_label == 'FX_right'):
                    continue
                
                np.random.seed(1)
                points_num = points_num + len(original_point_set)
                points_sum = points_sum + original_point_set.sum(0)
                fiber_data_list.append({'label': trk_label, 'local_fiber_id': fiber_num, 'data': point_set})
                fiber_id_2_original_fiber[fiber_num] = original_point_set

                fiber_num += 1

            points_mean = points_sum / points_num
            subject_based_data_dict['points_mean'] = points_mean
            subject_based_data_dict['fiber_points_list'] = fiber_data_list

            pkl_name = osp.join(save_path, '82_processed_data_{}.pkl'.format(downsample_rate))
            with open(pkl_name, 'wb') as f:
                pickle.dump(subject_based_data_dict, f, pickle.HIGHEST_PROTOCOL)
            
            pkl_name = osp.join(save_path, '82_fiber_id_2_original_fiber_{}.pkl'.format(downsample_rate))
            with open(pkl_name, 'wb') as f:
                pickle.dump(fiber_id_2_original_fiber, f, pickle.HIGHEST_PROTOCOL)

        tbar.update(1)

    tbar.close()


def check_duplicate_fiber(query_pc, gallery_streamlines):
    same_size = []

    for i, point_set in enumerate(gallery_streamlines):
        if len(query_pc) != len(point_set):
            continue
        same_size.append(point_set)

    query_pc = query_pc.reshape(-1)
    if len(same_size) > 0:
        same_size = np.array(same_size)
        same_size = same_size.reshape(len(same_size), query_pc.shape[0])

        search_result = (query_pc == same_size).all(1)
        if search_result.sum() > 0:
            return True, np.where(search_result==True)[0][0]

    return False, 0


def check_duplicate_name(person_dir_list, downsample_rate):
    relation = {key: [] for key, _ in HCP105_name2id.items()}

    for person_dir in person_dir_list:
        save_path = osp.join(dir_in, person_dir)
        
        if osp.isdir(save_path):
            pid = person_dir

            for tract_file in os.listdir(osp.join(dir_in, person_dir, 'tracts')):
                if not 'trk' in tract_file:
                    continue

                trk_path = osp.join(dir_in, person_dir, 'tracts', tract_file)
                trk_label = tract_file.split('.')[0]
                label_id = HCP105_name2id[trk_label]

                sl_file = nib.streamlines.load(trk_path)
                query_fibers = sl_file.streamlines
                
                for key in HCP105_name2id:
                    if HCP105_name2id[key] <= label_id:
                        continue
                    
                    trk_path = osp.join(dir_in, person_dir, 'tracts', '{}.trk'.format(key))
                    sl_file = nib.streamlines.load(trk_path)
                    gallery_fibers = sl_file.streamlines

                    print('{}:{} = {}:{}'.format(trk_label, key, len(query_fibers), len(gallery_fibers)))

                    for i in trange(len(query_fibers)):
                        is_duplicate, _ = check_duplicate_fiber(query_fibers[i], gallery_fibers)
                        if is_duplicate:
                            if not key in relation[trk_label]:
                                relation[trk_label].append(key)
                            if not trk_label in relation[key]:
                                relation[key].append(trk_label)
                            break
        
            print(relation)


def check_duplicate_num(person_dir_list, downsample_rate):
    check_list = [
        ['STR_left', 'T_PAR_left'], ['STR_left', 'T_PREC_left'], ['T_PREC_left', 'T_POSTC_left'], ['T_PREC_right', 'T_POSTC_right'],
        ['T_PREC_right', 'STR_right'], ['ST_PREC_right', 'ST_POSTC_right'], ['ST_PREC_right', 'ST_PAR_right'],
        ['T_PAR_left', 'T_POSTC_left'], ['T_PAR_right', 'T_POSTC_right'],
        ['ST_PAR_left', 'ST_POSTC_left'], ['ST_PAR_right', 'ST_POSTC_right'],
        ['ST_PREF_left', 'ST_FO_left'], ['ST_PREF_right', 'ST_FO_right']
    ]
    print(check_list)

    tbar = tqdm.tqdm(total=len(check_list))
    for todo_fiber in check_list:
        for person_dir in person_dir_list:
            save_path = osp.join(dir_in, person_dir)
            
            if osp.isdir(save_path):
                pid = person_dir
                data_list = []
                counter = [[0, 0], [0, 0]]

                for tract_file in todo_fiber:
                    tract_file = tract_file + '.trk'
                    trk_path = osp.join(dir_in, person_dir, 'tracts', tract_file)
                    sl_file = nib.streamlines.load(trk_path)
                    streamlines = sl_file.streamlines
                    data_list.append(streamlines)
                
                counter[0][0] = len(data_list[0])
                counter[1][0] = len(data_list[1])
                
                flag = False
                for i in range(2):
                    for j in range(len(data_list[i])):
                        is_duplicate, _ = check_duplicate_fiber(data_list[i][j], data_list[1-i])
                        if is_duplicate:
                            counter[i][0] -= 1
                            counter[i][1] += 1
                            flag = True
                if flag:
                    print('{} {}, {}: {}'.format(todo_fiber[0], todo_fiber[1], pid, counter))
            
        tbar.update(1)
    tbar.close()


def update_trk_files(person_dir_list, downsample_rate):
    tbar = tqdm.tqdm(total=len(person_dir_list))

    for person_dir in person_dir_list:
        save_path = osp.join(dir_in, person_dir)
        
        if osp.isdir(save_path):
            pid = person_dir
            data_list = {'A': [], 'B': []}

            for tract_file in os.listdir(osp.join(dir_in, person_dir, 'tracts')):
                trk_path = osp.join(dir_in, person_dir, 'tracts', tract_file)
                trk_label = tract_file.split('.')[0]

                if not trk_label in include_relation:
                    continue
                
                sl_file = nib.streamlines.load(trk_path)
                data_list['B'] = sl_file.streamlines
                ignore_list = []

                for sub_class in include_relation[trk_label]:
                    trk_path = osp.join(dir_in, person_dir, 'tracts', '{}.trk'.format(sub_class))
                    sl_file = nib.streamlines.load(trk_path)
                    data_list['A'] = sl_file.streamlines

                    new_data_list = {'A': [], 'AB': [], 'B': []}
                    
                    for i in range(len(data_list['A'])):
                        is_duplicate, index = check_duplicate_fiber(data_list['A'][i], data_list['B'])
                        if is_duplicate:
                            new_data_list['AB'].append(data_list['B'][index])
                            ignore_list.append(index)
                        else:
                            new_data_list['A'].append(data_list['A'][i])
                    
                    if len(new_data_list['A']) != 0:
                        pkl_name = osp.join(save_path, '{}.pkl'.format(sub_class))
                        with open(pkl_name, 'wb') as f:
                            pickle.dump(new_data_list['A'], f, pickle.HIGHEST_PROTOCOL)
                    
                    pkl_name = osp.join(save_path, '{}+{}.pkl'.format(trk_label, sub_class))
                    with open(pkl_name, 'wb') as f:
                        pickle.dump(new_data_list['AB'], f, pickle.HIGHEST_PROTOCOL)
                
                for i in range(len(data_list['B'])):
                    if i in ignore_list:
                        continue
                    new_data_list['B'].append(data_list['B'][i])

                pkl_name = osp.join(save_path, '{}.pkl'.format(trk_label))
                with open(pkl_name, 'wb') as f:
                    pickle.dump(new_data_list['B'], f, pickle.HIGHEST_PROTOCOL)
            
            tbar.update(1)

    tbar.close()


def cal_fa_distance(fa_points, method='centroid'):
    if method == 'centroid':
        fa_points_list, key_list = [], []
        for key in fa_points:
            fa_points_list.append(fa_points[key].mean(0))
            key_list.append(key)

        fa_points_list = np.array(fa_points_list)
        fa_points_list = fa_points_list.reshape(-1, 3)
        distA = pdist(fa_points_list, metric='euclidean')
        distB = squareform(distA)
        sorted_index = np.argsort(distB, axis=1)
        
        fa_index_dict = {i: 0 for i in range(len(fa_points_list))}

        current_index = 0
        for i in range(len(fa_points_list)):
            nearest_dist = distB[i, sorted_index[i][1]]
            print(nearest_dist)
            if nearest_dist < 5 and sorted_index[i][1] < i:
                fa_index_dict[i] = fa_index_dict[sorted_index[i][1]]
            else:
                fa_index_dict[i] = current_index
                current_index += 1

        for i in range(len(key_list)):
            print('{}: {}, {}, {}'.format(key_list[i], fa_index_dict[3*i], fa_index_dict[3*i+1], fa_index_dict[3*i+2]))

        return key_list, fa_index_dict, fa_points_list


def np_euclidean(x, y):
    return np.sqrt(np.sum((x - y)**2))


def is_reverse(point_a, point_b):
    if np_euclidean(point_a[0], point_b[-1]) + np_euclidean(point_a[-1], point_b[0]) < np_euclidean(point_a[0], point_b[0]) + np_euclidean(point_a[-1], point_b[-1]):
        return True

    return False


def add_cr_label(person_dir_list, downsample_rate):
    tbar = tqdm.tqdm(total=len(person_dir_list))
    fa_num = 5

    fx_list = [
        '912447', '887373', '680957', '922854', '984472'
    ]

    fa_points = {key: [] for key in HCP105_name2id}

    for person_dir in person_dir_list:
        save_path = osp.join(dir_in, person_dir)
        if osp.isdir(save_path):
            pid = person_dir
            if pid in fx_list:
                tbar.update(1)
                continue
            pkl_name = osp.join(save_path, '82_processed_data_{}.pkl'.format(downsample_rate))

            with open(pkl_name, 'rb') as f:
                subject_based_data_dict = pickle.load(f)

            for i in range(len(subject_based_data_dict['fiber_points_list'])):
                item = subject_based_data_dict['fiber_points_list'][i]
                points_mean = subject_based_data_dict['points_mean']
                fiber = item['data']
                fiber = fiber - points_mean
                trk_label = item['label']

                interval = (fiber.shape[0] - 1) / (fa_num - 1)
                keypoint_index = [a * interval for a in range(fa_num)]
                keypoint_index = [round(a) for a in keypoint_index]
                fa_points[trk_label].append(fiber[keypoint_index])

        tbar.update(1)

    tbar.close()

    for key in fa_points:
        fa_points[key] = np.concatenate(fa_points[key], axis=0)
        fa_points[key] = fa_points[key].reshape(-1, fa_num, 3)

    key_list, fa_index_dict, fa_points_list = cal_fa_distance(fa_points)

    pkl_name = osp.join(dir_in, '82_ar_data_{}.pkl'.format(downsample_rate))
    with open(pkl_name, 'wb') as f:
        pickle.dump({'key_list': key_list, 'fa_index_dict': fa_index_dict, 'fa_points_list': fa_points_list}, f, pickle.HIGHEST_PROTOCOL)


def clean_data_dir(person_dir_list):
    tbar = tqdm.tqdm(total=len(person_dir_list))

    for person_dir in person_dir_list:
        save_path = osp.join(dir_in, person_dir)
        if osp.isdir(save_path):
            subject_file_list = os.listdir(save_path)

            for file_name in subject_file_list:
                if 'tracts' in file_name:
                    exact_file_name = osp.join(save_path, file_name)
                    print(exact_file_name)
                    # import ipdb; ipdb.set_trace()
                    shutil.rmtree(exact_file_name)
                    # os.remove(exact_file_name)

        tbar.update(1)
    tbar.close()


def run_task(task_function, cpu_worker_num, all_pid, downsample_rate):
    if cpu_worker_num == 1:
        task_function(all_pid, downsample_rate)
    else:
        run_process(task_function, all_pid, cpu_worker_num, downsample_rate)


def fiber_distance_cal_efficient(set1, set2=None, num_points=20):
    set1 = set1.reshape(set1.shape[0], -1)   # set1 [N, 3*n_p]
    set2 = set2.reshape(set2.shape[0], -1) if set2 is not None else set1  # set2 [M, 3*n_p]
    set1_squ = (set1 ** 2).sum(1).view(-1, 1)  # set1_squ [N, 1]
    set2_t = torch.transpose(set2, 0, 1)   # set2_t [3*n_p, M]
    set2_squ = (set2 ** 2).sum(1).view(1, -1)   # set2_squ [1, M]
    dist = set1_squ + set2_squ - 2.0 * torch.mm(set1, set2_t)  # dist [N, M]
    # Ensure diagonal is zero if set1=set2
    if set2 is None:
       dist = dist - torch.diag(dist.diag())  
    dist = torch.sqrt(torch.clamp(dist, 0.0, np.inf))
    
    mean_dist = torch.div(dist, num_points)
    
    return mean_dist


def compute_fiber_distance(pid):
    save_path = osp.join(dir_in, pid)

    if osp.isdir(save_path):
        pkl_name = os.path.join(save_path, '82_subject_based_data_dict_1.pkl')

        with open(pkl_name, 'rb') as f:
            subject_based_data_dict = pickle.load(f)

        fiber_points_list = subject_based_data_dict['fiber_points_list']
        fiber_dict = {'ILF_left': [], 'UF_left': []}
        for item in fiber_points_list:
            if item['label'] in ['ILF_left', 'UF_left']:
                fiber_dict[item['label']].append(item['data'])

    set1 = torch.from_numpy(np.array(fiber_dict['ILF_left']))
    set2 = torch.from_numpy(np.array(fiber_dict['UF_left']))
    fiber_distance = fiber_distance_cal_efficient(set1, set2)
    keypoint_distance = fiber_distance_cal_efficient(set1[:, 0, :], set2[:, 0, :])


def save_point_set_as_vtk(dir_name, file_name, point_set):
    vtk_name = os.path.join(dir_name, file_name)

    with open(vtk_name, 'w') as f:
        f.write('# vtk DataFile Version 3.0\nFiber point_L_C\nASCII\nDATASET POLYDATA\n')

        fiber_num = len(point_set)
        fiber_length_list = [len(a['data']) for a in point_set]
        point_num = sum(fiber_length_list)
        f.write('POINTS {} float\n'.format(point_num))

        for i in range(fiber_num):
            points = point_set[i]['data']
            for j in range(fiber_length_list[i]):
                f.write('{:.6f} {:.6f} {:.6f}\n'.format(points[j][0], points[j][1], points[j][2]))

        f.write('LINES {} {}\n'.format(fiber_num, point_num+fiber_num))
        current_index = 0
        for i in range(fiber_num):
            temp_string = '{} '.format(fiber_length_list[i])
            for j in range(fiber_length_list[i]):
                temp_string += '{} '.format(current_index)
                current_index += 1
            temp_string = temp_string[:-1] + '\n'

            f.write(temp_string)


if __name__ == '__main__':
    args = get_args()
    dir_in = args.data
    downsample_rate = args.dp_rate
    cpu_worker_num = args.n_thread
    person_dir_list = os.listdir(dir_in)
    all_pid = []

    for person_dir in person_dir_list:
        save_path = osp.join(dir_in, person_dir)
        if osp.isdir(save_path):
            pid = person_dir
            all_pid.append(pid)
    
    with open(osp.join(dir_in, 'all_pid.pkl'), 'wb') as f:
        pickle.dump(all_pid, f, pickle.HIGHEST_PROTOCOL)

    if args.clean_data_dir:
        clean_data_dir(all_pid)

    if args.check_duplicate_num:
        run_task(check_duplicate_num, cpu_worker_num, all_pid, downsample_rate)

    if args.check_duplicate_name:
        run_task(check_duplicate_name, cpu_worker_num, all_pid, downsample_rate)

    if args.update_trk_files:
        run_task(update_trk_files, cpu_worker_num, all_pid, downsample_rate)

    if args.data_clean:
        run_task(data_clean, cpu_worker_num, all_pid, downsample_rate)
    
    if args.data_preprocess:
        run_task(data_preprocess, cpu_worker_num, all_pid, downsample_rate)

    if args.add_cr_label:
        run_task(add_cr_label, cpu_worker_num, all_pid, downsample_rate)
    
    if args.compute_fiber_distance:
        compute_fiber_distance('599469')