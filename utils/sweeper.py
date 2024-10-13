import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0, parentdir)

class Sweeper(object):
  '''
  This class generates a Config object and corresponding config dict
  given an index and the config file.
  '''
  def __init__(self, config_file):
    with open(config_file, 'r') as f:
      self.config_dicts = json.load(f)
    self.get_num_combinations_of_dict(self.config_dicts)

  def get_num_combinations_of_dict(self, config_dict):
    '''
    Get # of combinations for configurations in a config dict
    '''
    assert type(config_dict) == dict, 'Config file must be a dict!'
    num_combinations_of_dict = 1
    for _, values in config_dict.items():
      num_combinations_of_list = self.get_num_combinations_of_list(values)
      num_combinations_of_dict *= num_combinations_of_list
    config_dict['num_combinations'] = num_combinations_of_dict

  def get_num_combinations_of_list(self, config_list):
    '''
    Get # of combinations for configurations in a config list
    '''
    assert type(config_list) == list, 'Elements in a config dict must be a list!'
    num_combinations_of_list = 0
    for value in config_list:
      if type(value) == dict:
        if not('num_combinations' in value.keys()):
          self.get_num_combinations_of_dict(value)
        num_combinations_of_list += value['num_combinations']
      else:
        num_combinations_of_list += 1
    return num_combinations_of_list

  def generate_config_for_idx(self, idx):
    '''
    Generate a config dict for the index.
    Index is from 1 to # of conbinations.
    '''
    # Get config dict given the index
    cfg = self.get_dict_value(self.config_dicts, (idx-1) % self.config_dicts['num_combinations'])
    # Set config index
    cfg['config_idx'] = idx
    # Set number of combinations
    cfg['num_combinations'] = self.config_dicts['num_combinations']

    return cfg

  def get_list_value(self, config_list, idx):
    for value in config_list:
      if type(value) == dict:
        if idx + 1 - value['num_combinations'] <= 0:
          return self.get_dict_value(value, idx)
        else:
          idx -= value['num_combinations']
      else:
        if idx == 0:
          return value
        else:
          idx -= 1
  
  def get_dict_value(self, config_dict, idx):
    cfg = dict()
    for key, values in config_dict.items():
      if key == 'num_combinations':
        continue
      num_combinations_of_list = self.get_num_combinations_of_list(values)
      value = self.get_list_value(values, idx % num_combinations_of_list)
      cfg[key] = value
      idx = idx // num_combinations_of_list
    return cfg
  
  def print_config_dict(self, config_dict):
    cfg_json = json.dumps(config_dict, indent=2)
    print(cfg_json, end='\n')


def unfinished_index(exp, file_name='log.txt', runs=1, max_line_length=10000):
  '''
  Find unfinished config indexes based on the existence of time info in the log file
  '''
  # Read config files
  config_file = f'./configs/{exp}.json'
  sweeper = Sweeper(config_file)
  # Read a list of logs
  print(f'[{exp}]: ', end=' ')
  for i in range(runs * sweeper.config_dicts['num_combinations']):
    log_file = f'./logs/{exp}/{i+1}/{file_name}'
    try:
      with open(log_file, 'r') as f:
        # Get last line
        try:
          f.seek(-max_line_length, os.SEEK_END)
        except IOError:
          # either file is too small, or too many lines requested
          f.seek(0)
        last_line = f.readlines()[-1]
        # Get time info in last line
        try:
          t = float(last_line.split(' ')[-2])
        except:
          print(i+1, end=', ')
          continue
    except:
      print(i+1, end=', ')
      continue
  print()

if __name__ == "__main__":
  for agent_config in os.listdir('../configs/'):
    if not '.json' in agent_config:
      continue
    config_file = os.path.join('../configs/', agent_config)
    sweeper = Sweeper(config_file)
    print(f'Number of total combinations in {agent_config}:', sweeper.config_dicts['num_combinations'])