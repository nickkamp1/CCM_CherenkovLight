import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
plt.style.use('paper.mplstyle')
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from IPython import display
import uproot
import pandas as pd
import json,codecs

from AnalysisTools.MLE import DetectorModel,Dataset 


'''
Expects a ROOT file formated according to CCMDumpSimulation
Extracted using uproot
'''
class EventDisplay:
  
  def __init__(self,dataFileName,time_bins):
    self.data_uproot = uproot.open(dataFileName)
    self.keys = self.data_uproot.keys()
    self.MLE_dataset = Dataset(dataFileName,time_bins=time_bins)
    self.time_bins=time_bins
    self.detector_events = {}
    self.PMTpos = {}
    self.PMTuncoated = {}
    for evenno in range(2):
      event = data_pandas = self.MLE_dataset.GetDetectorEvent(evenno) 
      for r,c,x,y,z,coat in np.array(event[["HitRow","HitCol","HitPosX","HitPosY","HitPosZ","Coating"]]):
        self.PMTpos[(r,c)] = [x,y,z]
        self.PMTuncoated[(x,y,z)] = coat=='U'

  def PlotPMTHitsVsTime(self,eventno,PMTid,
                        stacked=False):
    
    key = self.keys[eventno]
    data_pandas = self.data_uproot[key].arrays(["HitRow","HitCol","HitTime"],library="pd")

    if isinstance(PMTid,tuple): PMTlist = [PMTid]
    else: PMTlist = PMTid

    histlist = []
    timelist = []
    for (r,c) in PMTlist:
      data_pandas_PMT = data_pandas.query('HitRow==@r and HitCol==@c')
      histlist.append(data_pandas_PMT["HitTime"])
      timelist+= list(data_pandas_PMT["HitTime"])
    
    if(stacked): plt.hist(histlist,bins=self.time_bins,stacked=True)
    else: plt.hist(timelist,bins=self.time_bins)
    plt.show()
  
  def PlotAllHitsVsTime(self,
                        eventno):
    
    key = self.keys[eventno]
    data_pandas = self.data_uproot[key].arrays(["HitTime"],library="pd")
    
    plt.hist(data_pandas["HitTime"],bins=self.time_bins)
    plt.show()
  
  def PlotCoatedVsUncoatedHitsVsTime(self,
                                     eventno):
    
    key = self.keys[eventno]
    data_pandas = self.data_uproot[key].arrays(["HitTime","HitCoat"],library="pd")
    data_coat = data_pandas.query("HitCoat==0")
    data_uncoat = data_pandas.query("HitCoat==1")
    
    plt.hist(data_coat["HitTime"],bins=self.time_bins,label="Coated",histtype='step')
    plt.hist(data_uncoat["HitTime"],bins=self.time_bins,label="Uncoated",histtype='step')
    plt.legend()
    plt.show()

  def GetEmptyHitsDict(self,mode=''):
    hits = {}
    for (r,c) in self.PMTpos.keys():
      if mode=='sides':
        if r==0 or r==6: continue
      if mode=='top':
        if r!=0: continue
      if mode=='bottom':
        if r!=6: continue
      hits[(r,c)] = 0
    return hits

  def GetCoatedUncoatedScatterDataTopBottom(self,
                                            data,
                                            DetReco=False,
                                            Top=True):
    hits = self.GetEmptyHitsDict(mode='top' if Top else 'bottom')

    for r,c,d in np.array(data[["HitRow","HitCol","Detected"]]):
      #if (r,c) not in hits.keys(): hits[(r,c)] = 0
      if DetReco and not d: continue
      hits[(r,c)]+=1

    xlist_c,xlist_u = [],[]
    ylist_c,ylist_u = [],[]
    clist_c,clist_u = [],[]
    for (r,c) in hits.keys():
      x,y,z = self.PMTpos[(r,c)]
      if self.PMTuncoated[(x,y,z)]:
        xlist_u.append(y)
        ylist_u.append(-x if Top else x)
        clist_u.append(hits[(r,c)])
      else:
        xlist_c.append(y)
        ylist_c.append(-x if Top else x)
        clist_c.append(hits[(r,c)])
    
    return xlist_u,ylist_u,clist_u,xlist_c,ylist_c,clist_c
  
  def GetCoatedUncoatedScatterDataSide(self,
                                       data,
                                       DetReco=False):
    hits = self.GetEmptyHitsDict(mode='sides')
    
    for r,c,d in np.array(data[["HitRow","HitCol","Detected"]]):
      #if (r,c) not in hits.keys(): hits[(r,c)] = 0
      if DetReco and not d: continue
      hits[(r,c)]+=1

    xlist_c,xlist_u = [],[]
    ylist_c,ylist_u = [],[]
    clist_c,clist_u = [],[]
    for (r,c) in hits.keys():
      x,y,z = self.PMTpos[(r,c)]
      rad = np.sqrt(x**2 + y**2)
      theta = np.arctan(y/x)
      if(x<0 and y>0): theta += np.pi
      if(x<0 and y<0): theta -= np.pi 
      plotx = rad*theta
      if self.PMTuncoated[(x,y,z)]:
        xlist_u.append(plotx)
        ylist_u.append(z)
        clist_u.append(hits[(r,c)])
      else:
        xlist_c.append(plotx)
        ylist_c.append(z)
        clist_c.append(hits[(r,c)])
    
    return xlist_u,ylist_u,clist_u,xlist_c,ylist_c,clist_c

  def ResetEDaxes(self,ax):
    
    cmap = self.GetCmap()
    ax[0].clear()
    ax[1].clear()
    ax[2].clear()
    ax[3].clear()
    ax[0].set_facecolor(cmap(0))
    ax[1].set_facecolor(cmap(0))
    ax[2].set_facecolor(cmap(0))
    ax[0].set_xlim(-95,95)
    ax[0].set_ylim(-95,95)
    ax[1].set_xlim(-360,360)
    ax[1].set_ylim(-60,60)
    ax[2].set_xlim(-95,95)
    ax[2].set_ylim(-95,95)
    ax[3].set_xlim(-1,1)
    ax[3].set_ylim(-1,1)

  def GetCmap(cmapname=None,n=1001):
    # Set zero to grey in colormap
    current_map = plt.get_cmap(lut=n)
    new_map = current_map(np.linspace(0,1,n))
    new_map[0] = np.array([160/256, 160/256, 160/256, 1])
    return ListedColormap(new_map) 
  
  def UpdatePlotAllPMTsTimeSlice(self,
                                 ax,
                                 data_pandas,
                                 time_slice,
                                 size=200,
                                 vmax=20,
                                 DetReco=False):
    
    self.ResetEDaxes(ax)
    
    tmin,tmax = time_slice
    if DetReco:
      data_slice = data_pandas.query("HitTimeSmeared>@tmin and HitTimeSmeared<@tmax")
    else:
      data_slice = data_pandas.query("HitTime>@tmin and HitTime<@tmax")
    
    # Top PMTs
    data_top = data_slice.query("HitRow==0")
    xlist_u,ylist_u,clist_u,xlist_c,ylist_c,clist_c = self.GetCoatedUncoatedScatterDataTopBottom(data_top,DetReco=DetReco)
    ax[0].scatter(xlist_u,ylist_u,s=size,c=clist_u,edgecolors='black',linewidths=1,linestyle='--',vmin=0,vmax=vmax,cmap=self.GetCmap())
    ax[0].scatter(xlist_c,ylist_c,s=size,c=clist_c,edgecolors='black',linewidths=1,vmin=0,vmax=vmax,cmap=self.GetCmap())
    
    
    # Side PMTs
    data_side = data_slice.query("HitRow>0 and HitRow<6")
    xlist_u,ylist_u,clist_u,xlist_c,ylist_c,clist_c = self.GetCoatedUncoatedScatterDataSide(data_side,DetReco=DetReco)
    ax[1].scatter(xlist_u,ylist_u,s=size,c=clist_u,edgecolors='black',linewidths=1,linestyle='--',vmin=0,vmax=vmax,cmap=self.GetCmap())
    ax[1].scatter(xlist_c,ylist_c,s=size,c=clist_c,edgecolors='black',linewidths=1,vmin=0,vmax=vmax,cmap=self.GetCmap())
    
    # Bottom PMTs
    data_bottom = data_slice.query("HitRow==6")
    xlist_u,ylist_u,clist_u,xlist_c,ylist_c,clist_c = self.GetCoatedUncoatedScatterDataTopBottom(data_bottom,DetReco=DetReco,Top=False)
    ax[2].scatter(xlist_u,ylist_u,s=size,c=clist_u,edgecolors='black',linewidths=1,linestyle='--',vmin=0,vmax=vmax,cmap=self.GetCmap())
    ax[2].scatter(xlist_c,ylist_c,s=size,c=clist_c,edgecolors='black',linewidths=1,vmin=0,vmax=vmax,cmap=self.GetCmap())
  
  def PlotAllPMTsTimeSlice(self,
                           eventno,
                           time_slice,
                           size=300,
                           n=3,
                           ProcessString=None,
                           SaveString=None,
                           vmax=20,
                           DetReco=False):

    key = self.keys[eventno]
    if eventno not in self.detector_events.keys():
      self.detector_events[eventno] = self.MLE_dataset.GetDetectorEvent(eventno) 
    data_pandas = self.detector_events[eventno]
    if(ProcessString is not None): 
      data_pandas = data_pandas.query("HitCreatorProcess==@ProcessString")

    fig = plt.figure(figsize=(3*n,3*n))
    ax = [plt.subplot2grid(shape=(n,n),loc=(0,1),colspan=1),
          plt.subplot2grid(shape=(n,n),loc=(1,0),colspan=3),
          plt.subplot2grid(shape=(n,n),loc=(2,1),colspan=1),
          plt.subplot2grid(shape=(n,n),loc=(0,2),colspan=1)]
    
    
    self.UpdatePlotAllPMTsTimeSlice(ax,data_pandas,time_slice,vmax=vmax,DetReco=DetReco)
    
    axes = fig.add_axes([0.67, 0.70, 0.22, 0.03])
    cb = mpl.colorbar.ColorbarBase(axes, orientation='horizontal', 
                                   norm=mpl.colors.Normalize(0, vmax),  # vmax and vmin
                                   label='Number of Hits')
    
    title = ProcessString if ProcessString is not None else 'All Photons'
    title += '\n{} $<$ t [ns] $<$ {}'.format(time_slice[0],time_slice[1])
    ax[3].axis('off')
    ax[3].text(-0.80,0.1,title)
    
    if(SaveString is not None): plt.savefig(SaveString)
    
    plt.show()

  def PlotAllPMTsTimeGif(self,
                         eventno,
                         time_width=2,
                         time_max=100,
                         size=300,
                         n = 3,
                         interval=100,
                         ProcessString=None,
                         SaveString=None,
                         Display=False):

    
    key = self.keys[eventno]
    data_pandas = self.data_uproot[key].arrays(["HitRow","HitPosX","HitPosY","HitPosZ","HitTime","HitCoat"],library="pd")
    data_pandas["HitCreatorProcess"] = list(self.data_uproot[key]["HitCreatorProcess"].array())[0]
    if(ProcessString is not None): 
      data_pandas = data_pandas.query("HitCreatorProcess==@ProcessString")
    time_bins = np.arange(0,time_max,time_width)
    
    
    fig = plt.figure(figsize=(3*n,3*n))
    ax = [plt.subplot2grid(shape=(n,n),loc=(0,1),colspan=1),
          plt.subplot2grid(shape=(n,n),loc=(1,0),colspan=3),
          plt.subplot2grid(shape=(n,n),loc=(2,1),colspan=1),
          plt.subplot2grid(shape=(n,n),loc=(0,0),colspan=1)]

    
    def AnimationFunction(frame):
      time_slice=(time_bins[frame],time_bins[frame+1])
      self.UpdatePlotAllPMTsTimeSlice(ax,data_pandas,time_slice)
      title = ProcessString if ProcessString is not None else ''
      title += '\n{} $<$ t [ns] $<$ {}'.format(time_bins[frame],time_bins[frame+1])
      ax[3].axis('off')
      ax[3].text(-0.50,0.1,title)
      

    
    anim_created = FuncAnimation(fig, AnimationFunction, frames=len(time_bins)-1, interval=interval)

    if Display:
      video = anim_created.to_html5_video()
      html = display.HTML(video)
      display.display(html)

    if(SaveString is not None): anim_created.save(SaveString,dpi=300)
    
    plt.close()
    
    return anim_created

  def CalculateAvgTemplate(self,
                           nMax=np.inf,
                           ProcessString=None,
                           template_string=None):
    
    if template_string:
      pmt_pos = json.loads(codecs.open('/home/nwkamp/Research/CCM/CherenkovLight/AverageTemplates/10000_event_runs/pmt_pos_template.json','r',encoding='utf-8').read())
      pmt_coat = json.loads(codecs.open('/home/nwkamp/Research/CCM/CherenkovLight/AverageTemplates/10000_event_runs/pmt_coat_template.json','r',encoding='utf-8').read())
      avg_hit_template = json.loads(codecs.open(template_string,'r',encoding='utf-8').read())
    else: pmt_pos,avg_hit_template,pmt_coat = self.MLE_dataset.GetAverageResponse(nMax=nMax,ProcessString=ProcessString)
    
    def GetXY(x,y,z,loc):
      if loc=='sides':
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan(y/x)
        if(x<0 and y>0): theta += np.pi
        if(x<0 and y<0): theta -= np.pi 
        return r*theta,z
      elif loc=='top':
        return y,-x
      elif loc=='bottom':
        return y,x
    
    self.avg_xlist_coat = {'top':[],
                           'sides':[],
                           'bottom':[]}
    self.avg_ylist_coat = {'top':[],
                           'sides':[],
                           'bottom':[]}
    self.avg_clist_coat = {'top':[[] for _ in range(len(self.time_bins)-1)],
                           'sides':[[] for _ in range(len(self.time_bins)-1)],
                           'bottom':[[] for _ in range(len(self.time_bins)-1)]}
    self.avg_xlist_uncoat = {'top':[],
                             'sides':[],
                             'bottom':[]}
    self.avg_ylist_uncoat = {'top':[],
                             'sides':[],
                             'bottom':[]}
    self.avg_clist_uncoat = {'top':[[] for _ in range(len(self.time_bins)-1)],
                             'sides':[[] for _ in range(len(self.time_bins)-1)],
                             'bottom':[[] for _ in range(len(self.time_bins)-1)]}
    for pmt_key,rate in avg_hit_template.items():
      r = int(pmt_key[-1])
      if r==0: loc='top'
      elif r==6: loc='bottom'
      else: loc='sides'
      x,y = GetXY(*pmt_pos[pmt_key],loc=loc)
      if pmt_coat[pmt_key]:
        self.avg_xlist_coat[loc].append(x)
        self.avg_ylist_coat[loc].append(y)
        for i,val in enumerate(rate): self.avg_clist_coat[loc][i].append(val)
      else:
        self.avg_xlist_uncoat[loc].append(x)
        self.avg_ylist_uncoat[loc].append(y)
        for i,val in enumerate(rate): self.avg_clist_uncoat[loc][i].append(val)
    return avg_hit_template
  
  def PlotAvgTemplate(self,
                      time_bin,
                      size=200,
                      n=3,
                      ProcessString=None,
                      SaveString=None,
                      vmax=20):
  
    
    fig = plt.figure(figsize=(3*n,3*n))
    ax = [plt.subplot2grid(shape=(n,n),loc=(0,1),colspan=1),
          plt.subplot2grid(shape=(n,n),loc=(1,0),colspan=3),
          plt.subplot2grid(shape=(n,n),loc=(2,1),colspan=1),
          plt.subplot2grid(shape=(n,n),loc=(0,2),colspan=1)]
     
    
    if isinstance(time_bin,int):
      time_bin = [time_bin]

    self.ResetEDaxes(ax)
    for i,loc in enumerate(['top','sides','bottom']):
      scat_col = np.zeros(np.array(self.avg_clist_coat[loc]).shape[-1])
      for tb in time_bin:
        scat_col += np.array(self.avg_clist_coat[loc])[tb,:] 
      ax[i].scatter(self.avg_xlist_coat[loc],self.avg_ylist_coat[loc],c=scat_col,s=size,vmin=0,vmax=vmax,edgecolors='black',linewidths=1,cmap=self.GetCmap())
    for i,loc in enumerate(['top','sides','bottom']):
      scat_col = np.zeros(np.array(self.avg_clist_uncoat[loc]).shape[-1])
      for tb in time_bin:
        scat_col += np.array(self.avg_clist_uncoat[loc])[tb,:] 
      ax[i].scatter(self.avg_xlist_uncoat[loc],self.avg_ylist_uncoat[loc],c=scat_col,s=size,vmin=0,vmax=vmax,edgecolors='black',linewidths=1,linestyle='--',cmap=self.GetCmap())
  
    axes = fig.add_axes([0.67, 0.70, 0.22, 0.03])
    cb = mpl.colorbar.ColorbarBase(axes, orientation='horizontal', 
                                   norm=mpl.colors.Normalize(0, vmax),  # vmax and vmin
                                   label='Number of Hits')
      
    title = ProcessString if ProcessString is not None else 'All Photons'
    title += '\n{} $<$ t [ns] $<$ {}'.format(self.time_bins[time_bin[0]],self.time_bins[time_bin[-1]+1])
    ax[3].axis('off')
    ax[3].text(-0.80,0.1,title)

    if(SaveString is not None): plt.savefig(SaveString)
    
    plt.show()
    
    






  
    
    
    
    
