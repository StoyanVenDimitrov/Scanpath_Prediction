"""Train script.
Usage:
  tester.py <hparams> <dataset_root> <checkpoint_dir> [--cuda=<id>]
  tester.py -h | --help

Options:
  -h --help     Show this screen.
  --cuda=<id>   id of the cuda device [default: 0].
"""
import os, json
import torch
import numpy as np
from tqdm import tqdm
from docopt import docopt
from os.path import join
from dataset import process_data
from irl_dcb.config import JsonConfig
from torch.utils.data import DataLoader
from irl_dcb.models import LHF_Policy_Cond_Small
from irl_dcb.environment import IRL_Env4LHF
from irl_dcb import metrics
from irl_dcb import utils
torch.manual_seed(42619)
np.random.seed(42619)


def gen_scanpaths(generator,
                  env_test,
                  test_img_loader,
                  hparams,
                  num_sample=10):
    patch_num = hparams.Data.patch_num
    max_traj_len = hparams.Data.max_traj_length
    all_actions = []
    for i_sample in range(num_sample):
        progress = tqdm(test_img_loader,
                        desc='trial ({}/{})'.format(i_sample + 1, num_sample))
        for i_batch, batch in enumerate(progress):
            env_test.set_data(batch)
            img_names_batch = batch['img_name']
            cat_names_batch = batch['cat_name']
            with torch.no_grad():
                env_test.reset()
                trajs = utils.collect_trajs(env_test,
                                            generator,
                                            patch_num,
                                            max_traj_len,
                                            is_eval=True,
                                            sample_action=True)
                all_actions.extend([(cat_names_batch[i], img_names_batch[i],
                                     'present', trajs['actions'][:, i])
                                    for i in range(env_test.batch_size)])

    scanpaths = utils.actions2scanpaths(all_actions, patch_num, hparams.Data.im_w, hparams.Data.im_h)
    utils.cutFixOnTarget(scanpaths, bbox_annos)

    return scanpaths


if __name__ == '__main__':
    args = docopt(__doc__)
    device = torch.device('cuda:{}'.format(args['--cuda']))
    hparams = args["<hparams>"]
    dataset_root = args["<dataset_root>"]
    checkpoint = args["<checkpoint_dir>"]
    hparams = JsonConfig(hparams)
    bbox_annos = np.load(join(dataset_root, 'bbox_annos.npy'),
                         allow_pickle=True).item()
    with open(join(dataset_root,
                   'human_scanpaths_TP_trainval_train.json')) as json_file:
        human_scanpaths_train = json.load(json_file)
        
    # ! coco test data instead of validation set
    with open(join(dataset_root,
                   'coco_test.json')) as json_file:
        human_scanpaths_test = json.load(json_file)

    human_scanpaths_test = list(
            filter(lambda x: x['correct'] == 1, human_scanpaths_test))

    for scanpath in human_scanpaths_test:
        scanpath['X'] = [x * 512/1680 for x in scanpath['X']]
        scanpath['Y'] = [x * 320/1050 for x in scanpath['Y']]

    # dir of pre-computed beliefs
    DCB_dir_HR = join(dataset_root, 'DCBs/HR/')
    DCB_dir_LR = join(dataset_root, 'DCBs/LR/')
    data_name = '{}x{}'.format(hparams.Data.im_w, hparams.Data.im_h)

    # process fixation data
    dataset = process_data(human_scanpaths_train, human_scanpaths_test,
                           DCB_dir_HR,
                           DCB_dir_LR,
                           bbox_annos,
                           hparams)
    img_loader = DataLoader(dataset['img_valid'],
                            batch_size=64,
                            shuffle=False,
                            num_workers=16)
    print('num of test images =', len(dataset['img_valid']))

    # load trained model
    input_size = 134  # number of belief maps
    task_eye = torch.eye(len(dataset['catIds'])).to(device)
    generator = LHF_Policy_Cond_Small(hparams.Data.patch_count,
                                      len(dataset['catIds']), task_eye,
                                      input_size).to(device)
    state = torch.load(join(checkpoint, 'trained_generator.pkg'), map_location=device)
    generator.load_state_dict(state['model'])

    # generator.load_state_dict(
    #     torch.load(join(checkpoint, 'trained_generator.pkg'),
    #                map_location=device))

    generator.eval()

    # build environment
    env_test = IRL_Env4LHF(hparams.Data,
                           max_step=hparams.Data.max_traj_length,
                           mask_size=hparams.Data.IOR_size,
                           status_update_mtd=hparams.Train.stop_criteria,
                           device=device,
                           inhibit_return=True)

    # generate scanpaths
    print('sample scanpaths (10 for each testing image)...')
    predictions = gen_scanpaths(generator,
                                env_test,
                                img_loader,
                                hparams,
                                num_sample=10)
    
    # compute multimatch
    res = metrics.compute_mm(human_scanpaths_test, predictions, hparams.Data.im_w, hparams.Data.im_h)
    print('Multimatch done: ', res)