import argparse
import json
import mne
import mock
import numpy as np
import os
import pandas as pd
import pywt
import scipy.stats as sp
import sys
import xmltodict

from scipy import signal
from spectrum import arburg
from tqdm import tqdm as tqdm

NCHAN = 32
CONFIG_PATH = 'config.json'


# %%
def getDataFiles(sessions_dir):
  session_folders, sessions = list(sorted(os.listdir(sessions_dir))), []
  for session_folder in session_folders:
    _path = os.path.join(sessions_dir, session_folder)
    if os.path.isdir(_path):
      node = {'folder': session_folder}
      for subfile in os.listdir(_path):
        if subfile.endswith('.bdf'):
          node['bdf'] = subfile
        elif subfile.endswith('.xml'):
          node['xml'] = subfile
      if 'bdf' in node and 'xml' in node and 'S_Trial' in node['bdf']:
        sessions.append(node)
  return sessions[:2]


# %%
def parse_args():
  parser = argparse.ArgumentParser(
      description='Extract features from EEG sessions')
  parser.add_argument(
      '-e',
      '--extract-only',
      action='store_true',
      help='Only extract each session into pickle')
  parser.add_argument(
      '-m',
      '--merge-only',
      action='store_true',
      help='Only merge all sessions into a single pickle')
  parser.add_argument(
      '-b',
      '--baseline',
      action='store_true',
      help='Subtract mean value of the respective channel from each channel')
  parser.add_argument(
      '-s',
      '--standardize',
      action='store_true',
      help='Standardize each channel')
  parser.add_argument(
      '-a', '--average', action='store_true', help='Average out all channels')
  parser.add_argument(
      '--all', action='store_true', help='Extract all possible combinations')
  parser.add_argument(
      '-w', '--winsize', default=3, type=float, help='Window Size in seconds')
  actions = vars(parser.parse_args())
  if actions['all']:
    actions['baseline'] = True
    actions['standardize'] = True
    actions['average'] = True

  conf = {'actions': actions}
  with open(CONFIG_PATH, 'r') as fp:
    conf['paths'] = json.load(fp)
  feat_types = ['r']
  if actions['baseline']:
    feat_types += ['b']
  if actions['standardize']:
    feat_types += ['s']
  if actions['baseline'] and actions['standardize']:
    feat_types += ['b_s']
  conf['feat_types'] = feat_types
  return conf


# %%
def coeff_var(epochs):
  return sp.variation(epochs, axis=2)


def kurtosis(epochs):
  return sp.kurtosis(epochs, axis=2)


#? Returns d1_mean, d1_max, d2_mean, d2_max
def diff(epochs):
  d1, d2 = np.diff(epochs, n=1, axis=2), np.diff(epochs, n=2, axis=2)
  return np.mean(
      d1, axis=2), np.max(
          d1, axis=2), np.mean(
              d2, axis=2), np.max(
                  d2, axis=2)


def skew(epochs):
  return sp.skew(epochs, axis=2)


def ar_burg(epochs):

  model_order = 3

  def ar(row):
    v1, _, _ = arburg(row, model_order)
    return v1

  def my_arburg(x):
    return np.apply_along_axis(lambda x: x.real, 0, ar(x))

  abc = np.apply_along_axis(my_arburg, axis=2, arr=epochs)
  return [abc[:, :, i] for i in range(model_order)]


def hjorth(epochs):
  d1 = np.diff(epochs, axis=2)
  d2 = np.diff(epochs, axis=2, n=2)
  h_activity = np.var(epochs, axis=2)
  h_mobility = np.sqrt(np.var(d1, axis=2) / h_activity)
  h_mobility_diff = np.sqrt(np.var(d2, axis=2) / np.var(d1, axis=2))
  h_complexity = h_mobility_diff / h_mobility
  return h_activity, h_mobility, h_complexity


def max_power_welch(epochs, sfreq):
  BandF = [0.1, 3, 7, 12, 30]
  f, Psd = signal.welch(
      epochs,
      sfreq,
  )
  return [
      np.max(
          Psd[:, :, np.where((f > BandF[i]) & (f <= BandF[i + 1]))].squeeze(),
          axis=2) for i in range(len(BandF) - 1)
  ]


def wavelet_features(epochs, nchan=NCHAN):
  cA, cD = pywt.dwt(epochs, 'coif1')

  def w_mean(c):
    return np.mean(c, axis=2)

  def w_std(c):
    return np.std(c, axis=2)

  def w_energy(c):
    return np.sum(np.square(c), axis=2)

  def w_entropy(c):
    return np.sum(np.square(c) * np.log(np.square(c)), axis=2)

  feats = [w_mean, w_std, w_energy, w_entropy]
  return [feat(c) for c in [cA, cD] for feat in feats]


def standardize(nda):
  return (nda - np.min(nda, axis=-1, keepdims=True)) / (
      np.max(nda, axis=-1, keepdims=True) - np.min(nda, axis=-1, keepdims=True))


#%% #*The MahnobEEG class
class MahnobEEG:

  def __init__(self, conf, session_info):
    self.session_info = session_info
    self.sessions_dir = conf['paths']['sessions_dir']
    self.meta_file = f"{self.sessions_dir}/{self.session_info['folder']}/{self.session_info['xml']}"
    self.extract_metadata()
    self.eeg_file = f"{self.sessions_dir}/{self.session_info['folder']}/{self.session_info['bdf']}"
    self.session_folder = f"{self.sessions_dir}/{self.session_info['folder']}"
    self.channels = []
    self.conf = conf
    self.feat_types = conf['feat_types']

  def extract_metadata(self):
    emodims = [
        '@feltArsl', '@feltCtrl', '@feltEmo', '@feltPred', '@feltVlnc',
        '@isStim'
    ]
    # *Extract metadata into meta
    temp = None
    with open(self.meta_file) as f:
      temp = xmltodict.parse('\n'.join(f.readlines()))
    temp = json.loads(json.dumps(temp))['session']
    metadata = {
        'subjectid': temp['subject']['@id'],
        'results': {k[1:]: int(temp[k]) for k in emodims},
        'media': {
            'name': temp['@mediaFile'],
            'durationSec': float(temp['@cutLenSec'])
        }
    }
    self.metadata = metadata

  def extract_BDF(self):
    stdout_old, stderr_old = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = mock.MagicMock(), mock.MagicMock()

    raw = mne.io.read_raw_bdf(
        self.eeg_file, preload=True, stim_channel='Status')
    t20 = mne.channels.make_standard_montage(kind='biosemi32')
    raw.set_montage(t20, raise_if_subset=False)
    events = mne.find_events(raw, stim_channel='Status')
    start_samp, end_samp = events[0][0] + 1, events[1][0] - 1
    raw.crop(raw.times[start_samp], raw.times[end_samp])
    self.nchan = 32
    ch2idx = dict(
        map(lambda x: (x[1], x[0]), list(enumerate(raw.ch_names[:self.nchan]))))
    raw.pick_channels(raw.ch_names[:self.nchan])
    self.ch2idx = dict(
        map(lambda x: (x[1], x[0]), list(enumerate(raw.ch_names[:self.nchan]))))
    self.df = raw.to_data_frame().rename(columns=ch2idx).T

    sys.stdout, sys.stderr = stdout_old, stderr_old

    nda = self.df.to_numpy()
    nda_base, nda_std, nda_base_std = None, None, None
    actions = self.conf['actions']
    ndas = {'r': nda}
    if 'b' in self.feat_types:
      nda_base = nda - np.mean(nda, axis=-1, keepdims=True)
      ndas['b'] = nda_base
    if 'b_s' in self.feat_types:
      nda_base_std = standardize(nda_base)
      ndas['b_s'] = nda_base_std
    if 's' in self.feat_types:
      nda_std = standardize(nda)
      ndas['s'] = nda_std
    self.ndas = ndas
    # self.nda = self.nda - np.mean(self.nda, axis=-1, keepdims=True)
    self.nsamp = nda.shape[1]
    self.sfreq = int(raw.info['sfreq'])
    self.samp_step = int(self.sfreq * actions['winsize'])
    self.chunk_shape = (self.nchan, actions['winsize'] * self.sfreq)

  def extract_features(self, nda):
    split_idcs = [
        self.chunk_shape[1] * (i + 1)
        for i in range(nda.shape[1] // self.chunk_shape[1])
    ]
    epochs_arr = np.split(nda, split_idcs, axis=1)
    #? num_of_epochs * nchan * pts_per_win
    ep = np.stack(epochs_arr[:-1])

    fd = {}
    fd['coeff_var'] = coeff_var(ep)
    fd['kurtosis'] = kurtosis(ep)
    fd['skew'] = skew(ep)
    fd['d1_mean'], fd['d1_max'], fd['d2_mean'], fd['d2_max'] = diff(ep)
    fd['ar1'], fd['ar2'], fd['ar3'] = ar_burg(ep)

    h = 'hjworth_'
    fd[f'{h}activity'], fd[f'{h}mobility'], fd[f'{h}complexity'] = hjorth(ep)

    a, b, c, d = max_power_welch(ep, self.sfreq)
    pr, pm = 'PRatio', 'PMax'
    fd[f'{pm}1'], fd[f'{pm}2'], fd[f'{pm}3'], fd[f'{pm}4'] = a, b, c, d
    fd[f'{pr}1'], fd[f'{pr}2'], fd[f'{pr}3'], fd[
        f'{pr}4'] = a / b, a / c, b / d, (a + b) / c

    wvf_names = [
        f'{c}_{feat}' for c in ['cA', 'cD']
        for feat in ['mean', 'std', 'energy', 'entropy']
    ]
    wvf_values = wavelet_features(ep)
    for i, name in enumerate(wvf_names):
      fd[name] = wvf_values[i]

    actions = self.conf['actions']
    dfs = {'act': pd.DataFrame(), 'avg': pd.DataFrame()}
    df_names = ['act']
    if actions['average']:
      df_names += ['avg']

    for feat in fd.keys():
      if actions['average']:
        dfs['avg'][feat] = np.mean(fd[feat], axis=-1)
      for i in range(self.nchan):
        dfs['act'][f'ch_{i:02}_{feat}'] = fd[feat][:, i]

    for df_name in df_names:
      dfs[df_name]['valence'] = self.metadata['results']['feltVlnc']
      dfs[df_name]['arousal'] = self.metadata['results']['feltArsl']
      dfs[df_name]['control'] = self.metadata['results']['feltCtrl']
      dfs[df_name]['prediction'] = self.metadata['results']['feltPred']
      dfs[df_name]['emotion'] = self.metadata['results']['feltEmo']
      dfs[df_name]['stim_video'] = self.metadata['media']['name']
      dfs[df_name]['subjectid'] = self.metadata['subjectid']

    print(dfs['act'], dfs['avg'])
    return dfs['act'], dfs['avg']

  def process(self):
    actions = self.conf['actions']
    self.extract_metadata()
    self.extract_BDF()
    for k in self.feat_types:
      feat, feat_avg = self.extract_features(self.ndas[k])
      if actions['average']:
        feat_avg.to_pickle(f'{self.session_folder}/features_a_{k}.pkl')
      feat.to_pickle(f'{self.session_folder}/features_{k}.pkl')


# %%
def main():
  if not os.path.isfile(CONFIG_PATH):
    print('Please create config.json in the root directory')
  conf = parse_args()
  print(json.dumps(conf, indent=2))
  sessions = getDataFiles(conf['paths']['sessions_dir'])
  if not conf['actions']['merge_only']:
    print('Extracting features ...')
    for session_info in tqdm(sessions):
      pt = MahnobEEG(conf, session_info)
      pt.process()
  if not conf['actions']['extract_only']:
    ks = conf['feat_types'][:]
    if conf['actions']['average']:
      ks += [f'a_{k}' for k in ks]
    for k in ks:
      print(f'Merging features of type {k}...')
      df_merged = pd.DataFrame()
      for session_info in tqdm(sessions):
        feat_file = f"{conf['paths']['sessions_dir']}/{session_info['folder']}/features_{k}.pkl"
        each = pd.read_pickle(feat_file)
        df_merged = pd.concat([df_merged, each])
      print(f"Saving to {conf['paths']['feat_dir']}/features_{k}.pkl ...")
      df_merged.to_pickle(f"{conf['paths']['feat_dir']}/features_{k}.pkl")


if __name__ == "__main__":
  main()
