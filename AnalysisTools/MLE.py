import numpy as np
import uproot
import pandas as pd
from scipy.special import factorial

PMTmap_file = '/home/nwkamp/Research/CCM/CherenkovLight/CCMAnalysisPublic/Config/mapping_master_7-14-22.csv'
hc = 197.3 * 2 * np.pi #eV nm

def get_idx(t,time_bins):
  return np.argmax(t<time_bins)-1

def smear_time_PMT(t,rise_time=2.5):
  return t + np.random.normal(loc=rise_time/2,scale=rise_time/2)

def smear_time_TPB(t,TPB_time_const=1.7):
  return t + np.random.exponential(scale=TPB_time_const)

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
               data_pandas,
               seed=None):
    self.data_pandas = data_pandas
    self.pmt_map = pd.read_csv(PMTmap_file)
    self.data_pandas['HitWavelength'] = hc / self.data_pandas['HitEnergy']
    pmt_keys = []
    for r,c in zip(data_pandas['HitRow'],data_pandas['HitCol']):
      pmt_keys.append('C'+str(c)+'R'+str(r))
    self.data_pandas['Position'] = pmt_keys
    self.orig_keys = list(self.data_pandas.keys())
    self.seed = seed

  def SmearTime(self):
    if self.seed is not None: np.random.seed(self.seed)
    self.data_pandas['HitTimeSmeared'] = smear_time_PMT(self.data_pandas['HitTime'])
    self.data_pandas['HitTimeSmeared'] = np.where(self.data_pandas['Coating']=='U',
                                                  self.data_pandas['HitTimeSmeared'],
                                                  smear_time_TPB(self.data_pandas['HitTimeSmeared']))
    self.orig_keys += ['HitTimeSmeared']

  def FindCoatedPMTs(self):
    self.data_pandas = pd.merge(self.data_pandas,self.pmt_map,on="Position",how="inner")
    self.data_pandas = self.data_pandas[self.orig_keys + ['Coating']]

  # Default efficiencies from Janet
  # Assume different constant efficiency above/below 200 nm 
  def ApplyTPBeff(self,
                  TPBlow=0.4,
                  TPBhigh=0.66,
                  TPBgeo=0.4,
                  TPBvis=0.8):
    self.data_pandas['TPBeff'] = np.where(self.data_pandas['HitWavelength']>200,TPBhigh*TPBgeo,TPBlow*TPBgeo)
    self.data_pandas['TPBeff'] = np.where(self.data_pandas['HitWavelength']>400,self.data_pandas['TPBeff']*TPBvis,self.data_pandas['TPBeff'])
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
    self.FindCoatedPMTs()
    self.SmearTime()
    self.ApplyTPBeff()
    self.ApplyPMTeff()
    self.data_pandas['DetEff'] = self.data_pandas['PMTeff'] * self.data_pandas['TPBeff']
    

'''
Expects a ROOT file formated according to CCMDumpSimulation
Extracted using uproot
'''
class Dataset:
  
  def __init__(self,dataFileName,time_bins=np.linspace(0,10,6)):
    self.data_uproot = uproot.open(dataFileName)
    self.keys = self.data_uproot.keys()
    self.time_bins = time_bins
    self.detector_events = {}

  def GetDetectorEvent(self,
                       evenno,
                       seed=None):
    data_pandas = self.data_uproot[self.keys[evenno]].arrays(["HitPosX","HitPosY","HitPosZ","HitRow","HitCol","HitTime","HitEnergy"],library="pd")
    data_pandas["HitCreatorProcess"] = list(self.data_uproot[self.keys[evenno]]["HitCreatorProcess"].array())[0]
    DetModel = DetectorModel(data_pandas,seed=seed)
    DetModel.ApplyDetectorEffects()
    if seed is not None: np.random.seed(seed)
    det_rand_arr = np.random.uniform(size=len(DetModel.data_pandas))
    DetModel.data_pandas["Detected"] = det_rand_arr < DetModel.data_pandas["DetEff"]
    return DetModel.data_pandas
  
  def GetEventMap(self,
                  evenno,
                  ProcessString=None):
    if evenno not in self.detector_events.keys():
      self.detector_events[evenno] = self.GetDetectorEvent(evenno)
    data_pandas = self.detector_events[evenno]
    data_pandas = data_pandas.query("Detected==1")
    if ProcessString:
      data_pandas = data_pandas.query("HitCreatorProcess==@ProcessString")
    event_map = {}
    for pmt_key,r,c,t in np.array(data_pandas[["Position","HitRow","HitCol","HitTimeSmeared"]]):
      #pmt_key = (r,c)
      if pmt_key not in event_map.keys(): event_map[pmt_key] = np.zeros(len(self.time_bins)-1)
      event_map[pmt_key][get_idx(t,self.time_bins)] += 1
    return event_map


  
  def LogLikelihood(self,
                    evenno,
                    prob_one = None,
                    template=None,
                    ProcessString=None):
    if not template: template = self.avg_hit_template
    if not prob_one: prob_one = 1./len(self.keys)
    event_map = self.GetEventMap(evenno,ProcessString=ProcessString)
    LLH = 0
    for key,mu_arr in template.items():
      if key not in event_map: k_arr = np.zeros(len(self.time_bins)-1)
      else: k_arr = event_map[key]
      for k,mu in zip(k_arr,mu_arr):
        if mu <= 0: 
          mu = prob_one
        LLH += (-mu + k*np.log(mu) - np.log(factorial(k)))
    return LLH

  def GetAverageResponse(self,
                         nMax=np.inf,
                         ProcessString=None):
    
    self.N = min(nMax,len(self.keys))
    self.avg_hit_template = {}
    pmt_pos = {}
    pmt_coat = {}
    tmin,tmax = self.time_bins[0]-5,self.time_bins[-1]+5
    for i,key in enumerate(self.keys):
      print('%i out of %i'%(i,self.N),end='\r')
      if i > nMax: continue
      data_pandas = self.data_uproot[key].arrays(["HitPosX","HitPosY","HitPosZ","HitRow","HitCol","HitTime","HitEnergy"],library="pd")
      if(ProcessString is not None):
        data_pandas["HitCreatorProcess"] = list(self.data_uproot[key]["HitCreatorProcess"].array())[0]
        data_pandas = data_pandas.query("HitCreatorProcess==@ProcessString")
      data_pandas.query('HitTime>@tmin and HitTime<@tmax',inplace=True)
      DetModel = DetectorModel(data_pandas)
      DetModel.ApplyDetectorEffects()
      for x,y,z,pmt_key,r,c,t,e,coat in np.array(DetModel.data_pandas[["HitPosX","HitPosY","HitPosZ","Position","HitRow","HitCol","HitTimeSmeared","DetEff","Coating"]]):
        #pmt_key = (r,c)
        if pmt_key not in self.avg_hit_template.keys():
          pmt_pos[pmt_key] = [x,y,z]
          pmt_coat[pmt_key] = 0 if coat=='U' else 1
          self.avg_hit_template[pmt_key] = np.zeros(len(self.time_bins)-1)
        tbin = get_idx(t,self.time_bins)
        if(tbin!=-1): self.avg_hit_template[pmt_key][tbin] += e/self.N

    return pmt_pos,self.avg_hit_template,pmt_coat




