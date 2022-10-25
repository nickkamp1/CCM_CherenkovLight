import numpy as np
from AnalysisTools.EventDisplay import EventDisplay

time_bins=np.linspace(0,10,6)
ProcessString=None
nMax=1000

for E in [1,2,3,4,5,10]:
  electron_dllh = np.empty(nMax)
  proton_dllh = np.empty(nMax)
  eED = EventDisplay('SimulationOutputs/BatchOutput/electron_gun_E_%s_sim_output.root'%str(E),time_bins=time_bins)
  pED = EventDisplay('SimulationOutputs/BatchOutput/proton_gun_E_%s_sim_output.root'%str(E),time_bins=time_bins)
  e_template = eED.CalculateAvgTemplate(nMax=nMax,ProcessString=ProcessString,
                                        template_string='AverageTemplates/electron_gun_E_%s_template.json'%str(E))
  p_template = pED.CalculateAvgTemplate(nMax=nMax,ProcessString=ProcessString,
                                        template_string='AverageTemplates/proton_gun_E_%s_template.json'%str(E))
  print('\nStarting E = %s MeV'%str(E))
  for evenno in range(nMax):
    print('%s out of %s'%(evenno,nMax),end='\r')
    eLLH = eED.MLE_dataset.LogLikelihood(evenno,template=e_template,prob_one=1./nMax)
    pLLH = eED.MLE_dataset.LogLikelihood(evenno,template=p_template,prob_one=1./nMax)
    electron_dllh[evenno] = (eLLH-pLLH)
    eLLH = pED.MLE_dataset.LogLikelihood(evenno,template=e_template,prob_one=1./nMax)
    pLLH = pED.MLE_dataset.LogLikelihood(evenno,template=p_template,prob_one=1./nMax)
    proton_dllh[evenno] = (eLLH-pLLH)

  np.save('DeltaLLH_Data/electron_dLLH_E%s.npy'%str(E),electron_dllh)
  np.save('DeltaLLH_Data/proton_dLLH_E%s.npy'%str(E),proton_dllh)
