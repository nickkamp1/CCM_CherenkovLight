import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython import display
import uproot
import pandas as pd

from AnalysisTools.MLE import DetectorModel,Dataset,smear_time 


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

  def GetCoatedUncoatedScatterDataTopBottom(self,
                                            data,
                                            DetReco=False,
                                            Top=True):
    hits = {}
    uncoated = {}
    for x,y,c,d in np.array(data[["HitPosX","HitPosY","Coating","Detected"]]):
      if((x,y) not in hits):
        hits[(x,y)] = 0
      uncoated[(x,y)] = c=='U'
      if DetReco and not d: continue
      hits[(x,y)]+=1

    xlist_c,xlist_u = [],[]
    ylist_c,ylist_u = [],[]
    clist_c,clist_u = [],[]
    for (x,y) in hits.keys():
      if uncoated[(x,y)]:
        xlist_u.append(y)
        ylist_u.append(-x if Top else x)
        clist_u.append(hits[(x,y)])
      else:
        xlist_c.append(y)
        ylist_c.append(-x if Top else x)
        clist_c.append(hits[(x,y)])
    
    return xlist_u,ylist_u,clist_u,xlist_c,ylist_c,clist_c
  
  def GetCoatedUncoatedScatterDataSide(self,
                                       data,
                                       DetReco=False):
    hits = {}
    uncoated = {}
    for x,y,z,c,d in np.array(data[["HitPosX","HitPosY","HitPosZ","Coating","Detected"]]):
      if((x,y,z) not in hits):
        hits[(x,y,z)] = 0
      uncoated[(x,y,z)] = c=='U'
      if DetReco and not d: continue
      hits[(x,y,z)]+=1

    xlist_c,xlist_u = [],[]
    ylist_c,ylist_u = [],[]
    clist_c,clist_u = [],[]
    for (x,y,z) in hits.keys():
      r = np.sqrt(x**2 + y**2)
      theta = np.arctan(y/x)
      if(x<0 and y>0): theta += np.pi
      if(x<0 and y<0): theta -= np.pi 
      plotx = r*theta
      if uncoated[(x,y,z)]:
        xlist_u.append(plotx)
        ylist_u.append(z)
        clist_u.append(hits[(x,y,z)])
      else:
        xlist_c.append(plotx)
        ylist_c.append(z)
        clist_c.append(hits[(x,y,z)])
    
    return xlist_u,ylist_u,clist_u,xlist_c,ylist_c,clist_c

  def ResetEDaxes(self,ax):
    
    ax[0].clear()
    ax[1].clear()
    ax[2].clear()
    ax[3].clear()
    ax[0].set_xlim(-95,95)
    ax[0].set_ylim(-95,95)
    ax[1].set_xlim(-360,360)
    ax[1].set_ylim(-60,60)
    ax[2].set_xlim(-95,95)
    ax[2].set_ylim(-95,95)
    ax[3].set_xlim(-1,1)
    ax[3].set_ylim(-1,1)
    
  
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
    ax[0].scatter(xlist_u,ylist_u,s=size,c=clist_u,edgecolors='black',linewidths=1,linestyle='--',vmin=0,vmax=10)
    ax[0].scatter(xlist_c,ylist_c,s=size,c=clist_c,edgecolors='black',linewidths=1,vmin=0,vmax=vmax)
    
    
    # Side PMTs
    data_side = data_slice.query("HitRow>0 and HitRow<6")
    xlist_u,ylist_u,clist_u,xlist_c,ylist_c,clist_c = self.GetCoatedUncoatedScatterDataSide(data_side,DetReco=DetReco)
    ax[1].scatter(xlist_u,ylist_u,s=size,c=clist_u,edgecolors='black',linewidths=1,linestyle='--',vmin=0,vmax=vmax)
    ax[1].scatter(xlist_c,ylist_c,s=size,c=clist_c,edgecolors='black',linewidths=1,vmin=0,vmax=vmax)
    
    # Bottom PMTs
    data_bottom = data_slice.query("HitRow==6")
    xlist_u,ylist_u,clist_u,xlist_c,ylist_c,clist_c = self.GetCoatedUncoatedScatterDataTopBottom(data_bottom,DetReco=DetReco,Top=False)
    ax[2].scatter(xlist_u,ylist_u,s=size,c=clist_u,edgecolors='black',linewidths=1,linestyle='--',vmin=0,vmax=vmax)
    ax[2].scatter(xlist_c,ylist_c,s=size,c=clist_c,edgecolors='black',linewidths=1,vmin=0,vmax=vmax)
  
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
    data_pandas = self.MLE_dataset.GetDetectorEvent(eventno)
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
                           ProcessString=None):
    
    pmt_pos,avg_hit_template,pmt_coat = self.MLE_dataset.GetAverageResponse(nMax=nMax,ProcessString=ProcessString)
    
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
    for (r,c),rate in avg_hit_template.items():
      if r==0: loc='top'
      elif r==6: loc='bottom'
      else: loc='sides'
      x,y = GetXY(*pmt_pos[(r,c)],loc=loc)
      if pmt_coat[(r,c)]:
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
     
    
    self.ResetEDaxes(ax)
    for i,loc in enumerate(['top','sides','bottom']):
      ax[i].scatter(self.avg_xlist_coat[loc],self.avg_ylist_coat[loc],c=self.avg_clist_coat[loc][time_bin],s=size,vmin=0,vmax=vmax,edgecolors='black',linewidths=1)
    for i,loc in enumerate(['top','sides','bottom']):
      ax[i].scatter(self.avg_xlist_uncoat[loc],self.avg_ylist_uncoat[loc],c=self.avg_clist_uncoat[loc][time_bin],s=size,vmin=0,vmax=vmax,edgecolors='black',linewidths=1,linestyle='--')
  
    axes = fig.add_axes([0.67, 0.70, 0.22, 0.03])
    cb = mpl.colorbar.ColorbarBase(axes, orientation='horizontal', 
                                   norm=mpl.colors.Normalize(0, vmax),  # vmax and vmin
                                   label='Number of Hits')
      
    title = ProcessString if ProcessString is not None else 'All Photons'
    title += '\n{} $<$ t [ns] $<$ {}'.format(self.time_bins[time_bin],self.time_bins[time_bin+1])
    ax[3].axis('off')
    ax[3].text(-0.80,0.1,title)

    plt.show()
    
    






  
    
    
    
    
