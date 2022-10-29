import numpy as np
from AnalysisTools.EventDisplay import EventDisplay

time_bins=np.linspace(0,10,6)
nMax=1000

for E in [1,2,3,4,5,10]:
  all_dllh = np.empty(nMax)
  scint_dllh = np.empty(nMax)
  ED = EventDisplay('SimulationOutputs/BatchOutput/electron_gun_E_%s_sim_output.root'%str(E),time_bins=time_bins)
  all_template = ED.CalculateAvgTemplate(nMax=nMax,ProcessString=None,
                                         template_string='AverageTemplates/electron_gun_E_%s_AllPhotons_template.json'%(str(E)))
  scint_template = ED.CalculateAvgTemplate(nMax=nMax,ProcessString='Scintillation',
                                           template_string='AverageTemplates/electron_gun_E_%s_Scintillation_template.json'%(str(E)))
  print('\nStarting E = %s MeV'%str(E))
  for evenno in range(nMax):
    print('%s out of %s'%(evenno,nMax),end='\r')
    aLLH = ED.MLE_dataset.LogLikelihood(evenno,template=all_template,prob_one=1./nMax,ProcessString=None)
    sLLH = ED.MLE_dataset.LogLikelihood(evenno,template=scint_template,prob_one=1./nMax,ProcessString=None)
    all_dllh[evenno] = (aLLH-sLLH)
    aLLH = ED.MLE_dataset.LogLikelihood(evenno,template=all_template,prob_one=1./nMax,ProcessString='Scintillation')
    sLLH = ED.MLE_dataset.LogLikelihood(evenno,template=scint_template,prob_one=1./nMax,ProcessString='Scintillation')
    scint_dllh[evenno] = (aLLH-sLLH)

  np.save('DeltaLLH_Data/all_dLLH_E%s.npy'%str(E),all_dllh)
  np.save('DeltaLLH_Data/scint_dLLH_E%s.npy'%str(E),scint_dllh)
