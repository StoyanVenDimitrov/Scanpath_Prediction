from cmath import pi
import glob, os, json, argparse, torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from numpy import load


PATHS = {
    'vqa-mhug': ['data/mhug/mhug/vqa-mhug_gaze.pickle', 'data/mhug/mhug/vqa-mhug_bboxes.pickle'],
    'vqa-mhug-jr': ['mhug-jr/vqa-mhug-jr_gaze.pickle', 'mhug-jr/vqa-mhug-jr_bboxes.pickle'],
    'air-mhug': ['mhug/air-mhug_gaze.pickle', 'mhug/air-mhug_bboxes.pickle'],
    'air-mhug-jr': ['mhug-jr/air-mhug-jr_gaze.pickle', 'mhug-jr/air-mhug-jr_bboxes.pickle']
}

def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='VQA-MHUG Deliverable Generator Args')
    
    parser.add_argument('--DATASETS', 
                        dest='DATASETS',
                        nargs='+',
                        type=str,
                        choices=['vqa-mhug', 'vqa-mhug-jr', 'air-mhug', 'air-mhug-jr'],
                        required=True)
    
    parser.add_argument('--FORMATS', 
                        dest='FORMATS',
                        nargs='+',
                        type=str,
                        choices=['img-attmap', 'txt-attmap', 'scanpath', 'dataset'],
                        required=True)
    
    parser.add_argument('--OUT_PATH',
                        dest='OUT_PATH',
                        help='The output path (can be relative). The script will automatically create subdirs for each chosen dataset and format, but you should specify different output paths if running this script multiple times with different duration/normalization/size parameters',
                        type=str,
                        default='./deliverables')
    
    parser.add_argument('--DURATION_SCALED',
                        dest='DURATION_SCALED',
                        type=bool,
                        help='Whether the attention maps should be scaled by the fixation duration',
                        default=True)
    
    parser.add_argument('--NORMALIZE',
                        dest='NORMALIZE',
                        type=bool,
                        help='Whether the attention maps should be normalized to a distribution (sum to 1)',
                        default=False)
    
    parser.add_argument('--ATTMAP_SIZE',
                        dest='ATTMAP_SIZE',
                        help='If given, the IMAGE attention maps are scaled to it instead of given in the original stimulus size',
                        nargs='+',
                        type=int)
    
    args = parser.parse_args()
    return args

def makeTextHeatmap(fixations, bboxes, duration_scaled):
    '''
    Sum up durations of fixations per word bounding box.
    Returns sequence of fixation durations in same length as sentence.
    '''
    #fixations are dicts of x,y,duration
    #bboxes are list of TXT, IMG, [N word tokens] each a list of identifier, top, left, line_height, word_width (for image it's ymin, xmin, ymax, xmax)
    heatmap = np.zeros(len(bboxes)-2)
    for fix in fixations: 
        for t, (_, y, x, h, w) in enumerate(bboxes[2:]):
            if (x <= fix['x'] <= (x+w)) and (y <= fix['y'] <= (y+h)):
                if duration_scaled:
                    heatmap[t] += fix['duration']
                else:
                    heatmap[t] += 1
    return heatmap

def makeImageHeatmap(fixations, bboxes, duration_scaled=True):
    #fixations are dicts of x,y,duration
    #bboxes are list of TXT, IMG, [N word tokens] each a list of identifier, top, left, line_height, word_width (for image it's ymin, xmin, ymax, xmax)
    _, y_min, x_min, y_max, x_max = bboxes[1]
    width = int(x_max - x_min)
    height = int(y_max - y_min)
    heatmap = np.zeros((height, width))
    for fix in fixations: 
        if (x_min <= fix['x'] <= x_max) and (y_min <= fix['y'] <= y_max):
            x = int(fix['x'] - x_min)
            y = int(fix['y'] - y_min)
            gaussian = gaussian_heatmap(center=(x,y), image_size=(width, height), sig=(fix['ppd_x']/1.5, fix['ppd_y']/1.5))
            if duration_scaled:
                gaussian *= fix['duration']
            heatmap += gaussian # 120 pixel corresponds to 2 degree visual angle. A gaussian has about 3 sigma radius (6 sigma diameter) --> 20 px
            # here we choose smaller sigma (3 sigma diameter), maybe it is better to use 5 degree fovea and correct 6 sigma diameter, also visual acuity is an exponential function not gaussian (cmp. salicon)
    heatmap = heatmap/heatmap.max()
    return heatmap

def gaussian_heatmap(center=(2, 2), image_size=(10, 10), sig=(1,1)):
    """
    It produces single gaussian at expected center
    :param center:  the mean position (X, Y) - where high value expected
    :param image_size: The total image size (width, height)
    :param sig: The sigma value
    :return:
    """
    x_axis = np.linspace(0, image_size[0]-1, image_size[0]) - center[0]
    y_axis = np.linspace(0, image_size[1]-1, image_size[1]) - center[1]
    xx, yy = np.meshgrid(x_axis, y_axis)
    kernel = np.exp(-0.5 * (np.square(xx) / np.square(sig[0]) + np.square(yy) / np.square(sig[1])))
    return kernel

def downsample(attmap, size=(14, 14)):
    '''
    Scale down attmap to 14x14 pixel
    '''
    attmap = torch.tensor(attmap).unsqueeze(0).unsqueeze(0)
    attmap = F.interpolate(attmap, size=size, mode="bilinear", align_corners=False)
    attmap = attmap.squeeze(0).squeeze(0)
    return attmap.numpy()

def normalize(attmap):
    '''
    Normalize sum of attmap (numpy) to 1
    '''
    attmap_sum = attmap.sum()
    return attmap/attmap_sum if attmap_sum > 0 else attmap

def makeScanpath(fixations, bboxes, include_breaks=True, for_dataset=False):
    _, y_min, x_min, y_max, x_max = bboxes[1]
    width = (x_max - x_min)/512 if for_dataset else x_max - x_min
    height = (y_max - y_min)/320 if for_dataset else y_max - y_min
    scanpath = {"X": [], "Y": [], "T": []} if for_dataset else []
    for fix in fixations: 
        if (x_min <= fix['x'] <= x_max) and (y_min <= fix['y'] <= y_max):
            if for_dataset:
                scanpath["X"].append((fix['x']-x_min)/width)  
                scanpath["Y"].append((fix['y']-y_min)/height)  
                scanpath["T"].append(fix['duration']) 
            else:
                scanpath.append({'x': (fix['x']-x_min)/width, 'y': (fix['y']-y_min)/height, 'duration': fix['duration'], 'pupil': fix['pupil']})
        elif include_breaks:
            scanpath.append(None) #to indicate a break in the scanpath due to fixation not on the image
        else:
            continue
    return scanpath

def makePath(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
    args = parse_args()

    dataset_to_json = []
    qid_to_iid_dict = json.load(open('data/mhug/qid_to_iid_correspondence.json', 'r'))
    bbox_pd = pd.read_csv('data/mhug/vqa_mhug_for_emma.csv', names=["id","bboxes","image_h", "image_w"])
    target_from_emb_mhug = json.load(open('data/mhug/data_label_word_emb_mhug.json', 'r'))

    for dataset in tqdm(args.DATASETS, desc='dataset'):
        gaze_data = pd.read_pickle(PATHS[dataset][0])
        bboxes_data = pd.read_pickle(PATHS[dataset][1])
        
        if 'jr' in dataset:
            img_query = txt_query = 'accurate_eye == eye & plate == "plate"'
        else:
            img_query = 'accurate_eye == eye & plate == "imgplate"'
            txt_query = 'accurate_eye == eye & plate == "txtplate"'
        
        for qid, pid in tqdm(gaze_data.reset_index(level=-1).index.unique(), desc='sample'):
            img_fixations = gaze_data.loc[qid, pid].query(img_query)[['x', 'y', 'ppd_x', 'ppd_y', 'duration', 'pupil']].to_dict(orient='records')
            txt_fixations = gaze_data.loc[qid, pid].query(txt_query)[['x', 'y', 'ppd_x', 'ppd_y', 'duration', 'pupil']].to_dict(orient='records')
            bboxes = bboxes_data.loc[qid][['token', 'ymin', 'xmin', 'ymax', 'xmax']].values.tolist()
            
            for form in args.FORMATS:
                path = os.path.join(args.OUT_PATH, dataset, form)
                makePath(path)
                if form == 'img-attmap':
                    attmap = makeImageHeatmap(img_fixations, bboxes, args.DURATION_SCALED)
                    if args.ATTMAP_SIZE:
                        attmap = downsample(attmap, args.ATTMAP_SIZE)
                    if args.NORMALIZE:
                        attmap = normalize(attmap)
                    np.save(f'{path}/q{qid}_p{pid}.npy', attmap)
                elif form == 'txt-attmap':
                    attmap = makeTextHeatmap(txt_fixations, bboxes, args.DURATION_SCALED)
                    if args.NORMALIZE:
                        attmap = normalize(attmap)
                    np.save(f'{path}/q{qid}_p{pid}.npy', attmap)
                elif form == 'scanpath':
                    scanpath = makeScanpath(img_fixations, bboxes)
                    json.dump(scanpath, open(f'{path}/q{qid}_p{pid}', 'w'))
                elif form == 'dataset':
                    scanpath = makeScanpath(img_fixations, bboxes, for_dataset=True, include_breaks=False)
                    scanpath["name"] = qid
                    scanpath["subject"] = pid
                    scanpath["length"] = len(scanpath["X"])
                    scanpath["condition"] = 'present'
                    # ---- get target bbox -----
                    iid = qid_to_iid_dict[qid]
                    all_bboxes_str = bbox_pd.loc[bbox_pd['id']==int(iid), 'bboxes'].iloc[0]
                    all_bboxes_split = all_bboxes_str.split('],')
                    # get target bbox indicies 
                    labels = target_from_emb_mhug[iid][qid]['labels']['classes']
                    target_indicies = [index for index, obj in enumerate(all_bboxes_split) if any(x in obj for x in labels)]
            
                    # get bbox file
                    pad_iid = 'COCO_val2014_' + (12-len(iid))*'0' + iid + '.jpg.npz'
                    data = load(f'data/mhug/butd/{pad_iid}')
                    lst = data.files
                    # ! get a single target!
                    scanpath["bbox"] = data['bbox'][target_indicies[0]].tolist()
                    image_h = data['image_h']
                    image_w = data['image_w']
                    # ! bbox might need to be scaled 
                    dataset_to_json.append(scanpath)
    if dataset_to_json:
        json.dump(dataset_to_json, open('dataset.json', 'w'), indent=4)