import numpy as np
import uproot
import pandas as pd

PMTmap_file = '/home/nwkamp/Research/CCM/CherenkovLight/CCMAnalysisPublic/Config/mapping_master_7-14-22.csv'
hc = 197.3 * 2 * np.pi #eV nm

def get_idx(t,time_bins):
  return np.argmax(t<time_bins)-1

def smear_time(t,sigma_t=2.6):
  st = -1*np.ones_like(t)
  while np.any(st < 0):
    st = np.where(st<0,np.random.normal(loc=t,scale=sigma_t),st)
  return st


'''
Expects a pandas dataframe with information on Hit positions, timing, and wavelength
expected keys: 
              'HitPosX',
              'HitPosY',
              'HitPosZ',
              'HitRow',
              'HitCol',
              'HitCoat',
              'HitEnergy',
              'HitAngle',
              'HitTime',
              'HitCreatorProcess'
'''
class DetectorModel:

  def __init__(self,
               data_pandas):
    self.data_pandas = data_pandas
    self.pmt_map = pd.read_csv(PMTmap_file)
    self.data_pandas['HitWavelength'] = hc / self.data_pandas['HitEnergy']
    pmt_keys = []
    for r,c in zip(data_pandas['HitRow'],data_pandas['HitCol']):
      pmt_keys.append('C'+str(c)+'R'+str(r))
    self.data_pandas['Position'] = pmt_keys
    self.orig_keys = list(self.data_pandas.keys())

  def SmearTime(self):
    self.data_pandas['HitTimeSmeared'] = smear_time(self.data_pandas['HitTime'])
    self.orig_keys += ['HitTimeSmeared']

  def FindCoatedPMTs(self):
    self.data_pandas = pd.merge(self.data_pandas,self.pmt_map,on="Position",how="inner")
    self.data_pandas = self.data_pandas[self.orig_keys + ['Coating']]

  # Default efficiencies from Janet
  # Assume different constant efficiency above/below 200 nm 
  def ApplyTPBeff(self,
                  TPBlow=0.4,
                  TPBhigh=0.66,
                  TPBgeo=0.4):
    self.data_pandas['TPBeff'] = np.where(self.data_pandas['HitWavelength']>200,TPBhigh*TPBgeo,TPBlow*TPBgeo)
    self.data_pandas['TPBeff'] = np.where(self.data_pandas['Coating']=='U',1,self.data_pandas['TPBeff'])

  # Default QE from Janet
  # 15% for cryo PMTs
  # For uncoated PMTs only consider a certain wavelength range
  def ApplyPMTeff(self,
                  PMTQE=0.15):
      
    self.data_pandas['PMTeff'] = np.where(np.logical_and(self.data_pandas['HitWavelength']>300,
                                                         self.data_pandas['HitWavelength']<650),
                                                         PMTQE,0)
    self.data_pandas['PMTeff'] = np.where(self.data_pandas['Coating']=='0',PMTQE,self.data_pandas['PMTeff'])
  
  def ApplyDetectorEffects(self):
    self.SmearTime()
    self.FindCoatedPMTs()
    self.ApplyTPBeff()
    self.ApplyPMTeff()
    self.data_pandas['DetEff'] = self.data_pandas['PMTeff'] * self.data_pandas['TPBeff']
    

'''
Expects a ROOT file formated according to CCMDumpSimulation
Extracted using uproot
'''
class Dataset:
  
  def __init__(self,dataFileName):
    self.data_uproot = uproot.open(dataFileName)
    self.keys = self.data_uproot.keys()

  def GetAverageResponse(self,
                         nMax=np.inf,
                         ProcessString=None,
                         time_bins=np.linspace(0,10,6)):
    
    self.N = min(nMax,len(self.keys))
    avg_hit_template = {}
    pmt_pos = {}
    pmt_coat = {}
    tmin,tmax = time_bins[0],time_bins[-1]
    for i,key in enumerate(self.keys):
      if i > nMax: continue
      data_pandas = self.data_uproot[key].arrays(["HitPosX","HitPosY","HitPosZ","HitRow","HitCol","HitTime","HitEnergy"],library="pd")
      if(ProcessString is not None):
        data_pandas["HitCreatorProcess"] = list(self.data_uproot[key]["HitCreatorProcess"].array())[0]
        data_pandas = data_pandas.query("HitCreatorProcess==@ProcessString")
      data_pandas.query('HitTime>@tmin and HitTime<@tmax',inplace=True)
      DetModel = DetectorModel(data_pandas)
      DetModel.ApplyDetectorEffects()
      for x,y,z,r,c,t,e,coat in np.array(DetModel.data_pandas[["HitPosX","HitPosY","HitPosZ","HitRow","HitCol","HitTimeSmeared","DetEff","Coating"]]):
        pmt_key = (r,c)
        if pmt_key not in avg_hit_template.keys():
          pmt_pos[pmt_key] = [x,y,z]
          pmt_coat[pmt_key] = 0 if coat=='U' else 1
          avg_hit_template[pmt_key] = np.zeros(len(time_bins)-1)
        tbin = get_idx(t,time_bins)
        if(tbin!=-1): avg_hit_template[pmt_key][tbin] += e/self.N

    return pmt_pos,avg_hit_template,pmt_coat




