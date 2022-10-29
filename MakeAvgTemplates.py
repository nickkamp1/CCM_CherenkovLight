from AnalysisTools.MLE import Dataset
import json,codecs
import numpy as np
nMax=1000

time_bins=np.linspace(0,10,6)

for E in [1,2,3,4,5,10]:
  particle='electron'
  #for particle in ['electron','proton']:
  for ProcessString in ['Cerenkov','OpWLS']:
    print('\n',E,particle)
    infile = 'SimulationOutputs/BatchOutput/%s_gun_E_%s_sim_output.root'%(particle,str(E))
    dset = Dataset(infile,time_bins=time_bins)
    pmt_pos,avg_hit_template,pmt_coat = dset.GetAverageResponse(nMax=nMax,ProcessString=ProcessString)
    for k1 in avg_hit_template.keys():
        avg_hit_template[k1] = avg_hit_template[k1].tolist()
    template_string = 'AverageTemplates/%s_gun_E_%s_%s_template.json'%(particle,str(E),ProcessString if ProcessString else 'AllPhotons')
    with codecs.open(template_string,"w",encoding='utf-8') as outfile:
      json.dump(avg_hit_template,outfile)
    if E==1 and particle=='electron' and ProcessString=='Scintillation':
      pos_string = 'AverageTemplates/pmt_pos_template.json'
      with codecs.open(pos_string,"w",encoding='utf-8') as outfile:
        json.dump(pmt_pos,outfile)
      coat_string = 'AverageTemplates/pmt_coat_template.json'
      with codecs.open(coat_string,"w",encoding='utf-8') as outfile:
        json.dump(pmt_coat,outfile)

