import json
import os
import numpy as np

# 66% of task have inout shapes equal <- these can be tried out of the box
# 32% of tasks have all shapes equal
# First Env sizes:

# Format of demonstration tasks [ ( 2d np.array of color name strings , ( line coord, column coord, color index ) ) ... ]
# This is the same as the normal tasks.

color_map = ('BLACK', 'WHITE', 'BLUE', 'GREEN', 'RED', 'YELLOW', 'ORANGE', 'PURPLE', 'BROWN', 'PINK')

def get_data_loader(directory='ARC-master/data/training', only_inout_equal=False):
  for filename in os.listdir(directory):
    with open(os.path.join(directory,filename),'r') as f:
      task = json.load(f)
      task_id = os.path.splitext(filename)[0]

      if not only_inout_equal or inout_equal(task['train']):
        task = {'train': task['train'],
                'test': task['test']}
        yield task_id, task

def inout_equal(dataset):
  return all(np.array(ex['input']).shape == np.array(ex['output']).shape for ex in dataset)

def map_list_of_examples(loe,assume_inout_equal=False):
  map_map = lambda m: np.array([[color_map[el] for el in line] for line in m], dtype=object)

  def ex_to_coordinate_exs(ex):
    input = map_map(ex['input'])
    if 'output' not in ex:
      if not assume_inout_equal:
        raise ValueError("Unknown output dimensions, not supported so far.")
      return [(input, (i, j, None)) for i in range(input.shape[0]) for j in
            range(input.shape[1])]
    output = np.array(ex['output'], dtype=np.int)
    return [(input, (i, j, output[i][j])) for i in range(output.shape[0]) for j in
            range(output.shape[1])]

  return [ex for exs in [ex_to_coordinate_exs(ex) for ex in loe] for ex in exs]

def get_task_demonstrations_loader(only_inout_equal,dir):
  tasks = get_data_loader(only_inout_equal=only_inout_equal,directory=dir)
  for task_id,task in tasks:
    yield task_id,task['train']

def get_task_test_loader(only_inout_equal,dir):
  tasks = get_data_loader(only_inout_equal=only_inout_equal,directory=dir)
  for task_id,task in tasks:
    yield task_id,task['test']

prefix = "ARCTask{}"

demonstrations_dict,tests_dict = None,None

def load_data(only_inout_equal=True,dir='ARC-master/data/training'):
  global demonstrations_dict,tests_dict
  demonstrations_dict = {prefix.format(i): d for i, d in get_task_demonstrations_loader(only_inout_equal,dir)}
  tests_dict = {prefix.format(i): d for i, d in get_task_test_loader(only_inout_equal,dir)}


def demonstrations(name):
  loe = demonstrations_dict[name]
  return map_list_of_examples(loe)

def tests(name, assume_inout_equal=True):
  loe = tests_dict[name]
  return map_list_of_examples(loe,assume_inout_equal)
