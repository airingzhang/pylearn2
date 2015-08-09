'''
Created on Aug 1, 2015

@author: ningzhang
'''

import os
import numpy as np

class DataWriter(object):
  """Class for writing lots of data to disk."""

  def __init__(self, names, output_dir, memory, numdim_list, datasize=None):
    """Initializes a Data Writer.
    Args:
      names: Names used to identify the different data components. Will be used
        as prefixes for the output files.
      output_dir: Directory where the data will be written.
      memory: Size of each output chunk.
      numdim_list: Number of dimensions in each data component.
      datasize: Total number of data vectors that will be written. Having this
        number helps to save memory.
    """
    typesize = 4  # Fixed for now.
    self.typesize = typesize
    self.names = names
    self.output_dir = output_dir
    if not os.path.isdir(output_dir):
      os.makedirs(output_dir)
    self.numdim_list = numdim_list
    self.data_len = len(names)
    assert self.data_len == len(numdim_list)
    numdims = sum(numdim_list)
    total_memory = GetBytes(memory)
    if datasize is not None:
      total_memory_needed = datasize * typesize * numdims
      total_memory = min(total_memory, total_memory_needed)
    self.buffer_index = [0] * self.data_len
    self.dump_count = [0] * self.data_len
    self.data_written = [0] * self.data_len
    self.max_dumps = []
    self.buffers = []
    for numdim in numdim_list:
      memory = (total_memory * numdim) / numdims
      numvecs = memory / (typesize * numdim)
      data = np.zeros((numvecs, numdim), dtype='float32')
      self.buffers.append(data)
      if datasize is not None:
        max_dump = datasize / numvecs
        if datasize % numvecs > 0:
          max_dump += 1
        self.max_dumps.append(max_dump)
      else:
        self.max_dumps.append(1)

  def AddToBuffer(self, i, data):
    """Add data into buffer i."""
    buf = self.buffers[i]
    buf_index = self.buffer_index[i]
    datasize = data.shape[0]
    assert datasize + buf_index <= buf.shape[0], 'Not enough space in buffer.'
    buf[buf_index:buf_index + datasize] = data
    self.buffer_index[i] += datasize

  def FreeSpace(self, i):
    """Return amount of free space left."""
    return self.buffers[i].shape[0] - self.buffer_index[i]

  def HasSpace(self, i, datasize):
    """Return True if buffer i has space to add datasize more vectors."""
    buf = self.buffers[i]
    buf_index = self.buffer_index[i]
    return buf.shape[0] > buf_index + datasize
  
  def IsFull(self, i):
    return not self.HasSpace(i, 0)

  def DumpBuffer(self, i):
    """Write the contents of buffer i to disk."""
    buf_index = self.buffer_index[i]
    if buf_index == 0:
      return
    buf = self.buffers[i]
    output_prefix = os.path.join(self.output_dir, self.names[i])
    output_filename = '%s-%.5d-of-%.5d' % (
      output_prefix, (self.dump_count[i] + 1), self.max_dumps[i])
    self.dump_count[i] += 1
    np.save(output_filename, buf[:buf_index])
    self.buffer_index[i] = 0
    self.data_written[i] += buf_index

  def SubmitOne(self, i, d):
    datasize = d.shape[0]
    free_space = self.FreeSpace(i)
    if datasize > free_space:
      self.AddToBuffer(i, d[:free_space])
    else:
      self.AddToBuffer(i, d)
    if self.IsFull(i):
      self.DumpBuffer(i)
    if datasize > free_space:
      self.SubmitOne(i, d[free_space:])

  def Submit(self, data):
    assert len(data) == self.data_len
    for i, d in enumerate(data):
      self.SubmitOne(i, d)

  def Commit(self):
    for i in range(self.data_len):
      self.DumpBuffer(i)
    return self.data_written
    
def GetBytes(mem_str):
  """Converts human-readable numbers to bytes.

  E.g., converts '2.1M' to 2.1 * 1024 * 1024 bytes.
  """
  unit = mem_str[-1]
  val = float(mem_str[:-1])
  if unit == 'G':
    val *= 1024 * 1024 * 1024
  elif unit == 'M':
    val *= 1024 * 1024
  elif unit == 'K':
    val *= 1024
  else:
    try:
      val = int(mem_str)
    except Exception:
      print '%s is not a valid way of writing memory size.' % mem_str
  return int(val) 
