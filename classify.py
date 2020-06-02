# Classify
import argparse
import json
import numpy as np
import os
import pandas as pd
import pickle
import random

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.svm import SVC
from xgboost import XGBClassifier


CONFIG_PATH = 'config.json'
MODELS = ['random_forest', 'xgboost']
MODELS_ACR = {'random_forest': 'rf', 'xgboost': 'xgb'}
# MODELS = ['random_forest', 'xgboost', 'svm_rbf']
DIVTYPES = ['random', 'subjectid']
exclude_labels = [
    'valence', 'arousal', 'control', 'prediction', 'emotion', 'stim_video',
    'subjectid', 'index'
  ]
emodim_labels = exclude_labels[:4]

def restricted_float(x):
  try:
    x = float(x)
  except ValueError:
    raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

  if x <= 0.0 or x >= 1.0:
    raise argparse.ArgumentTypeError("%r not in range (0.0, 1.0)" % (x,))
  return x

def get_datetime():
  return f'{datetime.now()}'.replace(' ', '_')

def parse_args():
  parser = argparse.ArgumentParser(
      description='Supervised classification of EEG features')
  parser.add_argument(
      '-b',
      '--baseline',
      action='store_true',
      help='Subtract mean value of the respective channel from each channel -- Use this feature'
  )
  parser.add_argument(
      '-s',
      '--standardize',
      action='store_true',
      help='Standardize each channel -- Use this feature')
  parser.add_argument(
      '-a',
      '--average',
      action='store_true',
      help='Average out all channels -- Use this feature')
  parser.add_argument(
      '-d',
      '--divtype',
      choices=DIVTYPES,
      default=['random'],
      nargs='+',
      help='How to split the data into train and test sets')
  parser.add_argument(
      '-m',
      '--model',
      choices=MODELS,
      default=[MODELS[0]],
      nargs='+',
      help='Model(s) to use for classification')
  parser.add_argument(
      '-r',
      '--random-state',
      type=int,
      default=1,
      help='Random State for Shuffle')
  parser.add_argument(
      '-tr',
      '--test-size',
      type=restricted_float,
      default=0.2,
      help='Test size in fraction (of total data points)')
  parser.add_argument(
      '-ts',
      '--test-subject-size',
      type=restricted_float,
      default=0.2,
      help='Test size in fraction (of number of subjects)')
  parser.add_argument(
      '-c',
      '--num-classes',
      type=int,
      choices=[2, 3, -1],
      default=3,
      help='Number of classes into which y will be divided')
  parser.add_argument(
      '--all',
      action='store_true',
      help='Use features from all possible combinations')
  actions = vars(parser.parse_args())
  actions['divtype'] = ['random']
  # if isinstance(actions['divtype'], str):
  #   actions['divtype'] = [actions['divtype']]
  if actions['all']:
    actions['baseline'] = True
    actions['standardize'] = True
    actions['average'] = True
    # actions['divtype'] = DIVTYPES[:]
    actions['model'] = MODELS[:]

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
  if actions['average']:
    feat_types += [f'a_{k}' for k in feat_types]
  conf['feat_types'] = feat_types
  return conf


def get_train_test(df, divtype, random_state, test_size, test_subject_size,
                   *args, **kwargs):
  random.seed(random_state)
  tt = {k: {} for k in divtype}
  df = df.sample(frac=1, random_state=random_state).dropna()
  if 'random' in divtype:
    test_size_n = int(test_size * len(df.index))
    tt['random']['test'] = df[:test_size_n]
    tt['random']['train'] = df[test_size_n:]
    print(f"Random train size : {tt['random']['train'].shape} Random test size : {tt['random']['test'].shape}")
  if 'subjectid' in divtype:
    subject_ids = df['subjectid'].unique()
    random.shuffle(subject_ids)
    test_subject_size_n = int(test_subject_size * len(subject_ids))
    test_subs = subject_ids[:test_subject_size_n]
    tt['subjectid']['test'] = df[df['subjectid'].isin(test_subs)]
    tt['subjectid']['train'] = df[~df['subjectid'].isin(test_subs)]
    print(f"Subjectid train size : {tt['subjectid']['train'].shape} Subjectid test size : {tt['subjectid']['test'].shape}")
  return tt

def get_X_Y(df, num_classes):
  felt_dim = {'min': 1, 'max': 9}
  def stratify_range(val):
    return ((val - felt_dim['min']) * num_classes) // (felt_dim['max'] - felt_dim['min'] + 1)
  X = df.drop(exclude_labels, axis=1)
  Y_raw = df[emodim_labels]
  Y = pd.DataFrame()
  for emodim_label in emodim_labels:
    Y[emodim_label] = Y_raw.apply(lambda row: stratify_range(row[emodim_label]), axis=1)
  return X, Y

def create_model_from_type(model_type):
  if model_type == 'random_forest':
    return RandomForestClassifier(n_estimators=50, max_depth=7)
  if model_type == 'xgboost':
    return XGBClassifier(n_estimators=50, max_depth=3, base_score=0.5)
  if model_type == 'svm_rbf':
    return SVC(kernel='rbf')

def classify(train_X, train_y, test_X, test_y, model_type, num_classes):
  train_X = np.asarray(train_X)
  train_y = np.asarray(train_y)
  test_X = np.asarray(test_X)
  test_y = np.asarray(test_y)
  model = create_model_from_type(model_type)
  model.fit(train_X, train_y)
  train_y_hat = model.predict(train_X)
  train_accuracy = accuracy_score(train_y, train_y_hat)
  test_y_hat = model.predict(test_X)
  test_accuracy = accuracy_score(test_y, test_y_hat)
  fpr, tpr, _ = roc_curve(test_y, pd.Series(test_y_hat), pos_label=num_classes - 1)
  return {'accuracy': {'train': train_accuracy, 'test': test_accuracy}, 'model': model, 'fpr': fpr, 'tpr': tpr}

def plot_feature_importance(feature_importances, file_path, show_fig=False):
  fig = make_subplots(rows=2, cols=2)
  for i in range(2):
    for j in range(2):
      ser = feature_importances[emodim_labels[i * 2 + j]].sort_values(ascending=False).nlargest(100)
      fig.add_trace(go.Bar(
          x=ser.keys(),
          y=ser,
          name=emodim_labels[i * 2 + j]
      ), row = i + 1, col = j + 1)
  fig.update_layout(title_text="Feature Importance", height=1000)
  fig.write_html(file_path)
  if show_fig:
    fig.show()
  else:
    return fig

def save_result(result, folder_path, prefix):
  if not os.path.isdir(folder_path):
    os.makedirs(folder_path)
  pickle.dump(result['model'], open(f'{folder_path}/model_{prefix}.sav', 'wb'))
  fp = open(f'{folder_path}/accuracy.md', 'a')
  str_to_write = f'''
  * {prefix}
    - Train Accuracy = {result['accuracy']['train']}
    - Test Accuracy  = {result['accuracy']['test']}
  '''
  print(str_to_write, file=fp)
  print(f'Saved {folder_path}/{prefix}')

def save_summary(folder_path, slugs,acc_train, acc_test):
  fig = go.Figure()
  fig.add_trace(go.Bar(x=slugs,
                  y=acc_train,
                  name='Train Accuracy',
                  marker_color='rgb(55, 83, 109)'
                  ))
  fig.add_trace(go.Bar(x=slugs,
                  y=acc_test,
                  name='Test Accuracy',
                  marker_color='rgb(26, 118, 255)'
                  ))

  fig.update_layout(
      title='Accuracy Summary',
      xaxis_tickfont_size=14,
      yaxis=dict(
          title='Accuracy',
          titlefont_size=16,
          tickfont_size=14,
      ),
      legend=dict(
          x=1.0,
          y=1.0,
          bgcolor='rgba(255, 255, 255, 0)',
          bordercolor='rgba(255, 255, 255, 0)'
      ),
      barmode='group',
      bargap=0.15, # gap between bars of adjacent location coordinates.
      bargroupgap=0.1 # gap between bars of the same location coordinate.
  )
  fig.write_html(f'{folder_path}/accuracy_summary.html')

def main():
  conf = parse_args()
  print(json.dumps(conf, indent=2))
  folder_name = get_datetime()
  folder_path = f"{conf['paths']['results_dir']}/{folder_name}"
  slugs = []
  acc_train = []
  acc_test = []
  for t in conf['feat_types']:
    feat_file = f"{conf['paths']['feat_dir']}/features_{t}.pkl"
    df = pd.DataFrame(pd.read_pickle(feat_file))
    tt = get_train_test(df, **conf['actions'])
    for e_div in tt.keys():
      train_X, train_Y = get_X_Y(tt[e_div]['train'], conf['actions']['num_classes'])
      test_X, test_Y = get_X_Y(tt[e_div]['test'], conf['actions']['num_classes'])
      for model_type in conf['actions']['model']:
        feature_importance_array = []
        common_prefix = f"{t}__{MODELS_ACR[model_type]}"
        roc_figure = go.Figure()
        for emodim_label in emodim_labels:
          train_y, test_y = train_Y[emodim_label], test_Y[emodim_label]
          result = classify(train_X, train_y, test_X, test_y, model_type, conf['actions']['num_classes'])
          prefix = f"{common_prefix}_{emodim_label[0]}"
          feature_importance_array.append(pd.Series(result['model'].feature_importances_, index=train_X.columns, name=emodim_label))
          roc_figure.add_trace(go.Scatter(x=result['fpr'], y=result['tpr'], mode='lines', name=emodim_label))
          save_result(result, folder_path, prefix)

          slugs.append(prefix)
          acc_train.append(result['accuracy']['train'])
          acc_test.append(result['accuracy']['test'])

        feature_importances = pd.concat(feature_importance_array, axis=1)
        plot_feature_importance(feature_importances, f'{folder_path}/{common_prefix}__feature_importances.html')
        roc_figure.write_html(f'{folder_path}/{common_prefix}__roc.html')
  
  save_summary(folder_path, slugs, acc_train, acc_test)



if __name__ == "__main__":
  main()
