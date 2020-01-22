import numpy as np
import arc_tasks

### Methods
def out_of_bounds(r, c, shape):
    return (r < 0 or c < 0 or r >= shape[0] or c >= shape[1])

def project_into_bounds(pos, shape):
  def clip(a, a_min, a_max):
    if a < a_min:
      return a_min
    elif a > a_max:
      return a_max
    else:
      return a
  return (clip(pos[0], 0, shape[0]), clip(pos[1], 0, shape[1]))

def shifted(direction, local_program, cell, obs):
    if cell is None:
        new_cell = None
    else:
        new_cell = (cell[0] + direction[0], cell[1] + direction[1])
        new_cell = project_into_bounds(new_cell, obs.shape)
        if len(cell) == 3:
          new_cell = new_cell + (cell[2],)
    return local_program(new_cell, obs)

def cell_is_value(value, cell, obs):
    if cell is None or out_of_bounds(cell[0], cell[1], obs.shape):
        focus = None
    else:
        focus = obs[cell[0], cell[1]]
    if value == 'ACTION_VALUE' and cell is not None:
      value = arc_tasks.color_map[cell[2]]

    return (focus == value)

def at_cell_with_value(value, local_program, cell, obs):
    if value == 'ACTION_VALUE':
      value = arc_tasks.color_map[cell[2]]
    matches = np.argwhere(obs == value)
    if len(matches) == 0:
        cell = None
    else:
        new_cell = tuple(matches[0])
        if len(cell) == 3:
          cell = new_cell + (cell[2],)
    return local_program(cell, obs)

def at_action_cell(local_program, cell, obs):
    return local_program(cell, obs)

def scanning(direction, true_condition, false_condition, cell, obs, max_timeout=50):
    if cell is None:
        return False

    for _ in range(max_timeout):
        current_cell = (cell[0] + direction[0], cell[1] + direction[1])
        if len(cell) == 3:
          current_cell = current_cell + (cell[2],)

        if true_condition(current_cell, obs):
            return True

        if false_condition(current_cell, obs):
            return False

        # prevent infinite loops
        if out_of_bounds(current_cell[0], current_cell[1], obs.shape):
            return False

    return False



### Grammatical Prior
START, CONDITION, LOCAL_PROGRAM, DIRECTION, POSITIVE_NUM, NEGATIVE_NUM, VALUE = range(7)

def create_grammar(object_types, arc_grammar):
    grammar = {
        START : ([['at_cell_with_value(', VALUE, ',', LOCAL_PROGRAM, ', a, s)'],
                  ['at_action_cell(', LOCAL_PROGRAM, ', a, s)']],
                  [0.5, 0.5]),
        LOCAL_PROGRAM : ([[CONDITION],
                          ['lambda cell,o : shifted(', DIRECTION, ',', CONDITION, ', cell, o)']],
                          [0.5, 0.5]),
        CONDITION : ([['lambda cell,o : cell_is_value(', VALUE, ', cell, o)'],
                      ['lambda cell,o : scanning(', DIRECTION, ',', LOCAL_PROGRAM, ',', LOCAL_PROGRAM, ', cell, o)']],
                      [0.5, 0.5]),
        DIRECTION : ([['(', POSITIVE_NUM, ', 0)'], ['(0,', POSITIVE_NUM, ')'],
                      ['(', NEGATIVE_NUM, ', 0)'], ['(0,', NEGATIVE_NUM, ')'],
                      ['(', POSITIVE_NUM, ',', POSITIVE_NUM, ')'], ['(', NEGATIVE_NUM, ',', POSITIVE_NUM, ')'],
                      ['(', POSITIVE_NUM, ',', NEGATIVE_NUM, ')'], ['(', NEGATIVE_NUM, ',', NEGATIVE_NUM, ')']],
                     [1./8] * 8),
        POSITIVE_NUM : ([['1'], [POSITIVE_NUM, '+1']],
                         [0.99, 0.01]),
        NEGATIVE_NUM : ([['-1'], [NEGATIVE_NUM, '-1']],
                         [0.99, 0.01]),
        VALUE : (object_types, 
                 [1./len(object_types) for _ in object_types])
    }
    if arc_grammar:
        print('Loading ARC grammar')
        grammar[VALUE] = (grammar[VALUE][0], [.5 if c == "'ACTION_VALUE'" else 1./(len(object_types)-1)/2. for c in grammar[VALUE][0]])
        print(grammar[VALUE])
    print(grammar)

    return grammar
    