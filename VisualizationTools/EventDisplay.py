import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython import display
import uproot
import pandas as pd

hc = 197.3 * 2 * np.pi #eV nm

'''
Expects a ROOT file formated according to CCMDumpSimulation
Extracted using uproot
'''
class EventDisplay:
  
  def __init__(self,dataFileName):
    self.data_uproot = uproot.open(dataFileName)
    self.keys = self.data_uproot.keys()

  def PlotPMTHitsVsTime(self,eventno,PMTid,
                        time_bins=np.linspace(0,20,100),
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
    
    if(stacked): plt.hist(histlist,bins=time_bins,stacked=True)
    else: plt.hist(timelist,bins=time_bins)
    plt.show()
  
  def PlotAllHitsVsTime(self,
                        eventno,
                        time_bins=np.linspace(0,20,100)):
    
    key = self.keys[eventno]
    data_pandas = self.data_uproot[key].arrays(["HitTime"],library="pd")
    
    plt.hist(data_pandas["HitTime"],bins=time_bins)
    plt.show()
  
  def PlotCoatedVsUncoatedHitsVsTime(self,
                                     eventno,
                                     time_bins=np.linspace(0,20,100)):
    
    key = self.keys[eventno]
    data_pandas = self.data_uproot[key].arrays(["HitTime","HitCoat"],library="pd")
    data_coat = data_pandas.query("HitCoat==0")
    data_uncoat = data_pandas.query("HitCoat==1")
    
    plt.hist(data_coat["HitTime"],bins=time_bins,label="Coated",histtype='step')
    plt.hist(data_uncoat["HitTime"],bins=time_bins,label="Uncoated",histtype='step')
    plt.legend()
    plt.show()

  def GetCoatedUncoatedScatterDataTopBottom(self,
                                            data):
    hits = {}
    uncoated = {}
    for x,y,c in np.array(data[["HitPosX","HitPosY","HitCoat"]]):
      if((x,y) not in hits):
        hits[(x,y)] = 0
      hits[(x,y)]+=1
      uncoated[(x,y)] = c

    xlist_c,xlist_u = [],[]
    ylist_c,ylist_u = [],[]
    clist_c,clist_u = [],[]
    for (x,y) in hits.keys():
      if uncoated[(x,y)]:
        xlist_u.append(x)
        ylist_u.append(y)
        clist_u.append(hits[(x,y)])
      else:
        xlist_c.append(x)
        ylist_c.append(y)
        clist_c.append(hits[(x,y)])
    
    return xlist_u,ylist_u,clist_u,xlist_c,ylist_c,clist_c
  
  def GetCoatedUncoatedScatterDataSide(self,
                                       data):
    hits = {}
    uncoated = {}
    for x,y,z,c in np.array(data[["HitPosX","HitPosY","HitPosZ","HitCoat"]]):
      if((x,y,z) not in hits):
        hits[(x,y,z)] = 0
      hits[(x,y,z)]+=1
      uncoated[(x,y,z)] = c

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

  def UpdatePlotAllPMTsTimeSlice(self,
                                 ax,
                                 data_pandas,
                                 time_slice,
                                 size=200):
    
    ax[0].clear()
    ax[1].clear()
    ax[2].clear()
    ax[0].set_title("Top")
    ax[1].set_title("Sides")
    ax[2].set_title("Bottom")
    ax[3].clear()
    ax[0].set_xlim(-95,95)
    ax[0].set_ylim(-95,95)
    ax[1].set_xlim(-360,360)
    ax[1].set_ylim(-60,60)
    ax[2].set_xlim(-95,95)
    ax[2].set_ylim(-95,95)
    ax[3].set_xlim(-1,1)
    ax[3].set_ylim(-1,1)
    
    tmin,tmax = time_slice
    data_slice = data_pandas.query("HitTime>@tmin and HitTime<@tmax")
    
    # Top PMTs
    data_top = data_slice.query("HitRow==6")
    xlist_u,ylist_u,clist_u,xlist_c,ylist_c,clist_c = self.GetCoatedUncoatedScatterDataTopBottom(data_top)
    ax[0].scatter(xlist_u,ylist_u,s=size,c=clist_u,edgecolors='black',linewidths=3,vmin=0,vmax=10)
    ax[0].scatter(xlist_c,ylist_c,s=size,c=clist_c,vmin=0,vmax=10)
    
    # Side PMTs
    data_side = data_slice.query("HitRow>0 and HitRow<6")
    xlist_u,ylist_u,clist_u,xlist_c,ylist_c,clist_c = self.GetCoatedUncoatedScatterDataSide(data_side)
    #ax[1].scatter(xlist_u,ylist_u,s=size,c=clist_u,edgecolors='black',linewidths=3)
    ax[1].scatter(xlist_u,ylist_u,s=size,c=clist_u,vmin=0,vmax=10)
    ax[1].scatter(xlist_c,ylist_c,s=size,c=clist_c,vmin=0,vmax=10)
    
    # Bottom PMTs
    data_bottom = data_slice.query("HitRow==0")
    xlist_u,ylist_u,clist_u,xlist_c,ylist_c,clist_c = self.GetCoatedUncoatedScatterDataTopBottom(data_bottom)
    ax[2].scatter(xlist_u,ylist_u,s=size,c=clist_u,edgecolors='black',linewidths=3,vmin=0,vmax=100)
    ax[2].scatter(xlist_c,ylist_c,s=size,c=clist_c,vmin=0,vmax=10)
  
  def PlotAllPMTsTimeSlice(self,
                           eventno,
                           time_slice,
                           size=300):

    key = self.keys[eventno]
    data_pandas = self.data_uproot[key].arrays(["HitRow","HitPosX","HitPosY","HitPosZ","HitTime","HitCoat"],library="pd")
    
    fig,ax = plt.subplots(3,1,gridspec_kw={'height_ratios':[1,1,1]},figsize=(6,18))
    
    self.UpdatePlotAllPMTsTimeSlice(ax,data_pandas,time_slice)
    
    plt.show()

  def PlotAllPMTsTimeGif(self,
                         eventno,
                         time_width=2,
                         time_max=100,
                         size=200,
                         n = 3,
                         interval=100):

    
    key = self.keys[eventno]
    data_pandas = self.data_uproot[key].arrays(["HitRow","HitPosX","HitPosY","HitPosZ","HitTime","HitCoat"],library="pd")
    time_bins = np.arange(0,time_max,time_width)
    
    
    fig = plt.figure(figsize=(3*n,3*n))
    ax = [plt.subplot2grid(shape=(n,n),loc=(0,1),colspan=1),
          plt.subplot2grid(shape=(n,n),loc=(1,0),colspan=3),
          plt.subplot2grid(shape=(n,n),loc=(2,1),colspan=1),
          plt.subplot2grid(shape=(n,n),loc=(0,0),colspan=1)]

    
    def AnimationFunction(frame):
      time_slice=(time_bins[frame],time_bins[frame+1])
      self.UpdatePlotAllPMTsTimeSlice(ax,data_pandas,time_slice)
      title = '{:.1f} < t < {:.1f}'.format(time_bins[frame],time_bins[frame+1])
      ax[3].axis('off')
      ax[3].text(-0.50,0,title)
      

    
    anim_created = FuncAnimation(fig, AnimationFunction, frames=len(time_bins)-1, interval=interval)

    video = anim_created.to_html5_video()
    html = display.HTML(video)
    display.display(html)

    plt.close()
    
    return anim_created



  
    
    
    
    
