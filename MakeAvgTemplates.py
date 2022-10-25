from AnalysisTools.MLE import Dataset
import json,codecs
import numpy as np

time_bins=np.linspace(0,10,6)

for E in [1,2,3,4,5,10]:
  for particle in ['electron','proton']:
    print('\n',E,particle)
    infile = 'SimulationOutputs/BatchOutput/%s_gun_E_%s_sim_output.root'%(particle,str(E))
    dset = Dataset(infile,time_bins=time_bins)
    pmt_pos,avg_hit_template,pmt_coat = dset.GetAverageResponse()
    for k1 in avg_hit_template.keys():
        avg_hit_template[k1] = avg_hit_template[k1].tolist()
    template_string = 'AverageTemplates/%s_gun_E_%s_template.json'%(particle,str(E))
    with codecs.open(template_string,"w",encoding='utf-8') as outfile:
      json.dump(avg_hit_template,outfile)
    if E==1 and particle=='electron':
      pos_string = 'AverageTemplates/pmt_pos_template.json'
      with codecs.open(pos_string,"w",encoding='utf-8') as outfile:
        json.dump(pmt_pos,outfile)
      coat_string = 'AverageTemplates/pmt_coat_template.json'
      with codecs.open(coat_string,"w",encoding='utf-8') as outfile:
        json.dump(pmt_coat,outfile)

