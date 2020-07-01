import numpy as np


def recursive_shape(obj, stop_length=1000):
  if isinstance(obj, (list, tuple)):
    shape = []
    containers_inside = False
    for el in obj:
      if isinstance(el, (list, tuple, dict)):
        containers_inside = True
        break
    if containers_inside:
      for el in obj:
        shape.append(recursive_shape(el, stop_length))
      if len(shape) > stop_length:
        shape = np.array(obj).shape 
    else:
      shape.append(len(obj))
  elif isinstance(obj, dict):
    containers_inside = False
    for el in obj.values():
      if isinstance(el, (list, tuple, dict)):
        containers_inside = True
        break
    if containers_inside:
      shape = {}
      for k, v in obj.items():
        shape[k] = recursive_shape(v, stop_length)
      if len(shape) > stop_length:
        shape = [len(shape)]
    else:
      shape = [len(obj)]
  elif isinstance(obj, np.ndarray):
    shape = obj.shape
  else:
    shape = None
  return shape


def recursive_type(obj, stop_length=1000):
  if isinstance(obj, (list, tuple)):
    types = []
    containers_inside = False
    for el in obj:
      if isinstance(el, (list, tuple, dict)):
        containers_inside = True
        break
    if containers_inside:
      for el in obj:
        types.append(recursive_type(el, stop_length))
      if len(types) > stop_length:
        types = type(obj)
    else:
      types.append(type(obj))
  elif isinstance(obj, dict):
    containers_inside = False
    for el in obj.values():
      if isinstance(el, (list, tuple, dict)):
        containers_inside = True
        break
    if containers_inside:
      types = {}
      for k, v in obj.items():
        types[k] = recursive_type(v, stop_length)
      if len(types) > stop_length:
        types = dict
    else:
      types = dict
  else:
    types = type(obj)
  return types
