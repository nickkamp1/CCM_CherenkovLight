import numpy as np

cm = 1
m = 1e-2
mm = 1e1
hc = 197 * np.pi * 2 # eV nm
eV = 1


lar_Energy_scin = [ 3.87*eV , 4.51*eV , 4.74*eV , 5.03*eV , 5.36*eV , 5.55*eV , 5.82*eV , 6.06*eV , 6.54*eV , 6.79*eV , 7.03*eV , 7.36*eV , 7.76*eV , 7.98*eV , 8.33*eV , 8.71*eV , 8.96*eV , 9.33*eV , 9.91*eV , 10.31*eV , 10.61*eV , 10.88*eV , 11.27*eV , 11.81*eV , 12.40*eV , 13.05*eV , 13.78*eV , 14.59*eV , 15.50*eV ] #energies for scintillation spectrum

lar_Energy_rin    = [ 1.771210*eV , 2.066412*eV , 2.479694*eV , 3.099618*eV ,
           4.132823*eV , 6.199235*eV , 6.888039*eV , 7.749044*eV ,
           8.856050*eV , 9.252590*eV , 9.686305*eV , 9.998766*eV ,
           10.33206*eV , 11.27134*eV , 12.39847*eV ] #/energies for refractive index and Rayleigh scattering lengths

lar_Energy_rs    = [ 1.771210*eV , 2.066412*eV , 2.479694*eV , 3.099618*eV ,
          4.132823*eV , 6.199235*eV , 6.888039*eV , 7.749044*eV ,
          8.856050*eV , 9.252590*eV , 9.686305*eV , 9.998766*eV ,
          10.33206*eV , 11.27134*eV , 12.39847*eV ] # energies for refractive index and Rayleigh scattering lengths*/

lar_SCINT = [ 0.00006, 0.00007, 0.00008, 0.00011, 0.00020, 0.00030, 0.00048, 0.00082, 0.00126, 0.00084, 0.00043, 0.00030, 0.00106, 0.00298, 0.00175, 0.00351, 0.01493, 0.12485, 0.49332, 0.20644, 0.07477, 0.04496, 0.01804, 0.00576, 0.00184, 0.00059, 0.00019, 0.00006, 0.00002 ]

lar_RIND  = [ 1.22 , 1.222 , 1.225 , 1.23 ,
         1.24 , 1.255 , 1.263 , 1.28 ,
         1.315, 1.335 , 1.358 , 1.403,
         1.45 , 1.62  , 1.79]

lar_RSL  = [ 327028.6808*cm, 172560.2267*cm, 80456.5339*cm, 31177.44642*cm,
        8854.144327*cm, 1496.876298*cm, 906.5011168*cm, 480.2538294*cm,
        205.3758714*cm, 145.6326111*cm, 100.7813004*cm, 63.2898117*cm,
        40.07450411*cm, 11.43903548*cm, 3.626432195*cm ]

def DefineLAr(# From DetectorConstruction.cc
            uvlas = 37.55,
            three = 1310.0,
            five = 12.2318,
            base = 55.9506,
            mult = 2800.0,
):
  time5 = five/100.;
  row5 = time5*base;
  uvlas5 = time5*uvlas;
  three5 = three;
  mult5 = mult;

  lar_Energy_abs = np.empty(47)
  lar_wlv_abs = np.empty(47)
  for i in range(47):
    lar_Energy_abs[i] = (1293.847/(i*20.0+80.0))*eV;
  lar_ABSL = np.empty(47)
  lar2_ABSL = np.empty(47)
  for i in range(6):
    lar_ABSL[i] = base*cm
    lar2_ABSL[i] = row5*cm
  for i in range(6,11):
    lar_ABSL[i] = uvlas*cm
    lar2_ABSL[i] = uvlas5*cm
  for i in range(11,16):
    lar_ABSL[i] = three*cm
    lar2_ABSL[i] = three5*cm
  for i in range(16,47):
    lar_ABSL[i] = mult*cm
    lar2_ABSL[i] = mult5*cm
  return lar_Energy_abs,lar_ABSL,lar2_ABSL


TPBEnergy = [ 0.602*eV,  0.689*eV,  1.030*eV,  1.926*eV, 2.138*eV,
      2.250*eV,  2.380*eV,  2.480*eV,  2.583*eV, 2.800*eV,
      2.880*eV,  2.980*eV,  3.124*eV,  3.457*eV, 3.643*eV,
      3.812*eV,  4.086*eV,  4.511*eV,  5.166*eV, 5.821*eV,
      6.526*eV,  8.266*eV,  9.686*eV,  11.27*eV, 12.60*eV]
    
TPBEmission =  [
      0.0000, 0.0000, 0.0000, 0.0000, 0.0005,
      0.0015, 0.0030, 0.0050, 0.0070, 0.0110,
      0.0110, 0.0060, 0.0020, 0.0000, 0.0000,
      0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
      0.0000, 0.0000, 0.0000, 0.0000, 0.0000
    ] #TPB Emission spectrum

TPBRIndex = [
      1.4, 1.4, 1.4, 1.4, 1.4,
      1.4, 1.4, 1.4, 1.4, 1.4,
      1.4, 1.4, 1.4, 1.4, 1.4,
      1.4, 1.4, 1.4, 1.4, 1.4,
      1.4, 1.4, 1.4, 1.4, 1.4
    ] #Refractive index of the TPB.
    
def DefineTpb(foil=.45548, pmt=0.96274, tpbAbs=0.8735):
  wlAbf21 = -0.002/(np.log((1-foil)))
  wlAbf24 = -0.002/(np.log((1-(.222*foil/0.2))))
  wlAbf19 = -0.002/(np.log((1-(.156*foil/0.2))))
  wlAbf15 = -0.002/(np.log((1-(.114*foil/0.2))))
  wlAbf12 = -0.002/(np.log((1-(.191*foil/0.2))))
  wlAbf11 = wlAbf21/(np.log(5.797944))
  TPBWLSAbsorption = [ 0.10000*m, 1000.000*m, 1000.000*m, 1000.000*m, 1000.000*m,
      1000.000*m, 1000.000*m, 1000.000*m, 1000.000*m, 1000.000*m,
      10000.000*m, 10000.000*m, 10000.000*m, 10000.000*m, 100000.0*m,
      100000.0*m, 100000.0*m, 100000.0*m, wlAbf24*mm, wlAbf21*mm,
      wlAbf19*mm, wlAbf15*mm, wlAbf12*mm, wlAbf11*mm, wlAbf11*mm ]
    
  lengthconst = 0.0019/(np.log(1-pmt));
  wlsAb110 = lengthconst/(np.log(0.067));
  wlsAb128 = lengthconst/(np.log(0.2));
  wlsAb190 = lengthconst/(np.log(0.533));
  wlsAb213 = lengthconst/(np.log(0.4));
  wlsAb240 = lengthconst/(np.log(0.367));
  TPBWLSAbsorption100 = [    0.10000*m, 1000.00*m, 1000.00*m, 1000.00*m, 1000.000*m,
   1000.00*m, 1000.00*m, 1000.00*m, 1000.00*m, 1000.000*m,
   10000.0*m, 10000.0*m, 10000.0*m, 10000.0*m, 100000.0*m,
   100000.0*m, 100000.0*m, 100000.0*m, wlsAb240*mm, wlsAb213*mm,
   wlsAb190*mm, wlsAb213*mm, wlsAb128*mm, wlsAb110*mm, wlsAb110*mm
    ]
  
  absltwo = -0.00211/(np.log(tpbAbs));
  abslttw = absltwo*1.4;
  abslttt = absltwo*1.55;
  TPBAbsorption = [  0.02000*mm, absltwo*mm, absltwo*mm, absltwo*mm, absltwo*mm,
       abslttw*mm, abslttw*mm, abslttw*mm, abslttw*mm, abslttw*mm,
       abslttt*mm, abslttt*mm, abslttt*mm, 10.0000*mm, 100000.0*m,
       100000.0*m, 100000.0*m, 100000.0*m, 100000.0*m, 100000.0*m,
       100000.0*m, 100.0000*m, 100.0000*m, 100.0000*m, 100.0000*m
    ]
    
  return TPBWLSAbsorption,TPBWLSAbsorption100,TPBAbsorption

  