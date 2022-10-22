import numpy as np
import uproot
import pandas as pd

def get_idx(t,time_bins):
  return np.argmax(t<time_bins)-1

'''
Expects a ROOT file formated according to CCMDumpSimulation
Extracted using uproot
'''
class Dataset:
  
  def __init__(self,dataFileName):
    self.data_uproot = uproot.open(dataFileName)
    self.keys = self.data_uproot.keys()
    self.N = len(self.keys)

  def GetAverageResponse(self,
                         time_bins=np.linspace(0,10,6)):
    
    avg_hit_template = {}
    pmt_pos = {}
    for key in self.keys():
      data_pandas = self.data_uproot[key].arrays(["HitPosX","HitPosY","HitPosZ","HitRow","HitCol","HitTime"],library="pd")
      for x,y,z,r,c,t in np.array(data_pandas[["HitPosX","HitPosY","HitPosZ","HitRow","HitCol","HitTime"]]):
        pmt_key = tuple(r,c)
        if pmt_key not in avg_hit_template.keys():
          pmt_pos[pmt_key] = [x,y,z]
          avg_hit_template[pmt_key] = np.zeros_like(time_bins)
        tbin = get_idx(t)
        if(tbin!=-1): avg_hit_template[pmt_key][tbin] += 1./self.N

    return pmt_pos,avg_hit_templat:




