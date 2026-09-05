# -------------------------------------------------------------------------
# Name:        Daily crop growth calculation
# Purpose:
#
# Author:      Zhonghao Fu
#
# Created:     23/09/2024
# Copyright:   (c) Zhonghao Fu 2024
# -------------------------------------------------------------------------

from laws.management_modules.globals import dateVar

import numpy as np
from laws.management_modules.data_handling import *
import warnings
from scipy.optimize import newton

class BioMassCalc(object):

    """
    Calculate daily crop phenology, canopy development, biomass, and yield.

    **Global variables**

    =====================================  ======================================================================  =========================
    Variable [self.var]                    Description                                                             Unit
    =====================================  ======================================================================  =========================
    Crops_names                            Names of crops represented by the crop-index dimension                  --
    FDM1                                   First encoded control point of the frost-damage response curve          --
    FDM2                                   Second encoded control point of the frost-damage response curve         --
    HCM                                    Maximum crop height                                                     m
    HIL                                    Lower harvest index under water-stressed reproductive growth            --
    HIO                                    Optimal harvest index                                                   --
    HUG                                    Heat units required for crop germination                                degree C day
    HUIS                                   Heat-unit fraction at which leaf-area decline begins                    --
    LAC1                                   First encoded control point of the leaf-area development curve          --
    LAC2                                   Second encoded control point of the leaf-area development curve         --
    LAIM                                   Maximum leaf area index                                                 m2 m-2
    LAR                                    Leaf-area-index decline rate                                            --
    EAct                                   Actual vapour pressure                                                  kPa
    GP_{Irr,nonIrr}                        Maximum growing-period duration for irrigated and rainfed crops          day
    GrowPeriod_{Irr,nonIrr}                Elapsed growing-period duration for irrigated and rainfed crops          day
    HDI                                    Input index used to select the maize harvest-index parameter            --
    HIA                                    Harvest index after crop-specific adjustment                            --
    HUF_{Irr,nonIrr}_daily                 Heat-unit factor for leaf-area development for irrigated and rainfed crops --
    HUI_{Irr,nonIrr}_daily                 Accumulated heat units divided by potential heat units for both systems --
    JJP_{Irr,nonIrr}                       Crop-emergence flag for irrigated and rainfed crops                      bool
    JJT_{Irr,nonIrr}                       Crop status for both systems: 0 absent, 1 growing, 2 mature, 3 harvest  int
    LAI_{Irr,nonIrr}_daily                 Leaf area per unit ground area for irrigated and rainfed crops          m2 m-2
    N_stress_{Irr,nonIrr}                  Nutrition-stress multiplier for irrigated and rainfed crops              --
    PHU_{Irr,nonIrr}                       Potential heat units to maturity for irrigated and rainfed crops        degree C day
    POP                                    Plant population density                                                plant m-2
    PPLP1                                  First encoded control point of the plant-population/LAI curve           --
    PPLP2                                  Second encoded control point of the plant-population/LAI curve          --
    RDM                                    Maximum rooting depth                                                  m
    RDR                                    Radiation-use decline rate                                             --
    RD_{Irr,nonIrr}_daily                  Daily root depth for irrigated and rainfed crops                        m
    REG_{Irr,nonIrr}_daily                 Biomass-growth stress multiplier for irrigated and rainfed crops        --
    Rsds                                   Surface-downwelling shortwave radiation used by the biomass routine     MJ m-2 day-1
    RTF1                                   Root fraction of total biomass at emergence                            --
    RTF2                                   Root fraction of total biomass at physiological maturity               --
    SCRP3                                  Drives harvest index development as a function of crop Maturity
    SCRP10                                 Calculates the effect of water stress on harvest index as
                                            Function of plant water use
    STLBm_{Irr,nonIrr}_daily               Above-ground biomass for irrigated and rainfed crops                    t ha-1
    TMax                                   Daily maximum air temperature                                           degree C
    TMin                                   Daily minimum air temperature                                           degree C
    Tavg                                   Daily mean air temperature                                              degree C
    totalBm_{Irr,nonIrr}_daily             Total biomass for irrigated and rainfed crops                           t ha-1
    TS_daily                               Daily temperature-stress multiplier                                     --
    WA                                     Radiation-use efficiency at the reference CO2 concentration            kg ha-1/(MJ m-2)
    RUC2                                   Encoded CO2 response point for radiation-use efficiency                 --
    VPR                                    VPD effect on radiation-use efficiency                                 kg ha-1/(MJ m-2)/kPa
    WS_{Irr,nonIrr}_daily                  Daily water-stress multiplier for irrigated and rainfed crops           --
    actTransForHI_{Irr,nonIrr}_daily       Actual transpiration accumulated for harvest response in both systems   m
    actYield_{Irr,nonIrr}                  Harvested crop yield for irrigated and rainfed crops                    t ha-1
    co2                                    Atmospheric carbon-dioxide concentration                                ppm
    dHUF_{Irr,nonIrr}_daily                Daily heat-unit-factor increment for irrigated and rainfed crops        --
    height_{Irr,nonIrr}_daily              Daily crop height for irrigated and rainfed crops                       m
    lat                                    Latitude of each active model cell                                      degree north
    TB                                     Crop-specific base growth temperature                                  degree C
    TO                                     Crop-specific optimum growth temperature                               degree C
    potTransForHI_{Irr,nonIrr}_daily       Potential transpiration accumulated for harvest response in both systems m
    soildepth12                            Combined depth of the upper two CWatM soil layers                        m
    =====================================  ======================================================================  =========================

    **Functions**
    """

    def __init__(self, model):
        self.model = model
        self.var = model.var

    def initial(self):
        self.var.lat = get_latitudes_1d()

        self.HRLT = globals.inZero.copy()


        self.SCRP3_1 = 50.10
        self.SCRP3_2 = 95.95
        X1,Y1 = self.scrp_to_xy(self.SCRP3_1)
        X2,Y2 = self.scrp_to_xy(self.SCRP3_2)
        self.SCRP3_1, self.SCRP3_2 = self.calculate_s_curve_params(X1,Y1,X2,Y2)

        self.SCRP10_1 = 10.05
        self.SCRP10_2 = 50.90
        X1,Y1 = self.scrp_to_xy(self.SCRP10_1)
        X2,Y2 = self.scrp_to_xy(self.SCRP10_2)
        self.SCRP10_1, self.SCRP10_2 = self.calculate_s_curve_params(X1,Y1,X2,Y2)

        self.LACB1 = np.zeros(len(self.var.Crops_names))
        self.LACB2 = np.zeros(len(self.var.Crops_names))
        self.FDMB1 = np.zeros(len(self.var.Crops_names))
        self.FDMB2 = np.zeros(len(self.var.Crops_names))
        self.RUC1 = np.zeros(len(self.var.Crops_names))
        self.RUC2 = np.zeros(len(self.var.Crops_names))
        self.PPLP1 = np.zeros(len(self.var.Crops_names))
        self.PPLP2 = np.zeros(len(self.var.Crops_names))
        self.RUE = np.zeros(len(self.var.Crops_names))

        for c in range(len(self.var.Crops_names)):

            X1,Y1 = self.scrp_to_xy(self.var.LAC1[c])
            X2,Y2 = self.scrp_to_xy(self.var.LAC2[c])
            X1 = X1/100
            X2 = X2/100
            self.LACB1[c], self.LACB2[c] = self.calculate_s_curve_params(X1,Y1,X2,Y2)

            X1,Y1 = self.scrp_to_xy(self.var.FDM1[c])
            X2,Y2 = self.scrp_to_xy(self.var.FDM2[c])
            self.FDMB1[c], self.FDMB2[c] = self.calculate_s_curve_params(X1,Y1,X2,Y2)

            Y1 = self.var.WA[c]/100
            X2,Y2 = self.scrp_to_xy(self.var.RUC2[c])
            self.RUC1[c], self.RUC2[c] = self.calculate_s_curve_params(330,Y1,X2,Y2)
            self.RUE[c] = 100.*self.var.co2/(self.var.co2+np.exp(self.RUC1[c]-self.var.co2*self.RUC2[c]))

            ##set plant popolation and max potential LAI
            X1,Y1 = self.scrp_to_xy(self.var.PPLP1[c])
            X2,Y2 = self.scrp_to_xy(self.var.PPLP2[c])
            self.PPLP1[c],self.PPLP2[c] = self.calculate_s_curve_params(X1,Y1,X2,Y2)
            if self.var.POP[c] == 0:

                G,G1 = self.scrp_to_xy(self.var.PPLP1[c])

                g1_initial_guess = G
                tolerance = 1e-5
                max_iterations = 10

                try:
                    g1_root = newton(
                        func=self.f,
                        x0=g1_initial_guess,
                        fprime=self.fprime,
                        args=(self.PPLP1[c], self.PPLP2[c]),
                        tol=tolerance,
                        maxiter=max_iterations
                    )
                    self.var.POP[c] = g1_root

                except RuntimeError as e:
                    # A RuntimeError is raised if the solver does not converge
                    # within the maximum number of iterations.
                    warnings.warn(f"!!!!! PLANT POP DID NOT CONVERGE after {max_iterations} iterations: {e}")
                    # When convergence fails, the Fortran implementation assigns
                    # G1 (the value at the final iteration) to X3. In contrast,
                    # scipy.optimize.newton does not return its final value after
                    # failure. Reproducing the Fortran behavior would therefore
                    # require a custom iteration loop or another solver that
                    # exposes the final iterate. Here, POP is set to NaN instead.
                    self.var.POP[c] = np.nan
                    raise CWATMError(
                        "Error 400: LAWS plant-population solver did not converge "
                        f"for crop {self.var.Crops_names[c]}: {e}"
                    )
            self.var.LAIM[c] = self.var.LAIM[c] * self.var.POP[c] / ( self.var.POP[c] + np.exp(self.PPLP1[c] - self.PPLP2[c] * self.var.POP[c]))

        self.YTN = np.tan(self.var.lat / 57.296)
        CH = 0.4349 * np.abs(self.YTN)

        self.HLMN = np.where(CH>=1,0,np.arccos(CH)*7.72)

        self.WDRM = np.where(self.HLMN<11,np.minimum(11,0.5+self.HLMN),self.HLMN)

        self.AJHU = np.where(self.HLMN<11,calculate_coldest_day_indices_2d_to_1d(),400)

        self.var.HIA = np.tile(globals.inZero, (len(self.var.Crops_names), 1))
        for c in range(len(self.var.Crops_names)):
            self.var.HIA[c] = self.var.HIO[c]
        if 'Maize' in self.var.Crops_names and 'HDI' in binding:
            self.var.HDI = loadmap('HDI')
            idx = np.where(np.array(self.var.Crops_names) == 'Maize')[0]
            self.var.HIA[idx] = np.where(self.var.HDI>=80,0.6,0.45)

    def calculateHUF(self,HUI,HUF0,c):
        """
        Ta: the daily mean air temperature
        Tb: the plant's specific base temperature
        Hun: the potential heat unit for the maturity of the crop
        Huna: the accumulation of the
        """
        #From emergence to the start of leaf decline
        HUF = HUI / (HUI + np.exp(self.LACB1[c] - self.LACB2[c] * HUI))
        dHUF = HUF - HUF0
        return HUF,dHUF


    def calculateDaylightHours(self):


        SD = .4102 * np.sin((dateVar['doy'] - 80.25) / 58.13)
        CH = -self.YTN * np.tan(SD)
        HRLT = np.where(CH>=1,0,np.where(CH<-1,24,np.arccos(CH)*7.72))

        if dateVar['newStart']:
            dayhr_diff = HRLT - self.HLMN
        else:
            dayhr_diff = HRLT - self.HRLT
        Frac = (dayhr_diff + 1)**3
        self.HRLT = HRLT
        return Frac

    def calculatedLAI(self,HUI,dHUF,LAI0,SLA0,WS,Frac,c):
        Frac = Frac * np.sqrt(WS)

        IXX =  dHUF * self.var.LAIM[c]
        DXX = np.maximum(np.log10((1-HUI)/(1-self.var.HUIS[c])),-5)

        SLAI = np.where(IXX>0,LAI0+IXX*Frac,LAI0)

        LAI = np.maximum(0.05,np.where((HUI > self.var.HUIS[c])&(self.HRLT>self.WDRM), np.minimum(SLA0 * 10 **(DXX*self.var.LAR[c]),SLAI),
                               SLAI))

        LAI = np.minimum(self.var.LAIM[c],LAI)

        return LAI

    def calculateBiomassDeclineFactor(self,HUI,c):
        """Calculate the AJWA biomass factor during leaf-area decline."""
        DXX = np.maximum(np.log10((1-HUI)/(1-self.var.HUIS[c])),-5)
        AJWA = np.where((HUI > self.var.HUIS[c]) & (self.HRLT > self.WDRM),10 ** (self.var.RDR[c] * DXX),1,)
        return AJWA

    def calculatedRD(self,HUI,RDM):
        RD = np.minimum(RDM,np.minimum(self.var.soildepth12+0.05,2.5*RDM*HUI))
        return RD

    def calculatedHeight(self,HUF,LAI,height0,HCM):
        height = np.where(LAI > 0.05,np.maximum(HCM * np.sqrt(HUF),height0),height0)
        return height

    def calculateBioMass(self,LAI,REG,BN,VPR,RUE,AJWA):
        ##only once###
        ESatmin = 0.6108* np.exp((17.27 * self.var.TMin) / (self.var.TMin + 237.3))
        ESatmax = 0.6108* np.exp((17.27 * self.var.TMax) / (self.var.TMax + 237.3))
        ESat = (ESatmin + ESatmax) / 2.0
        VPD = np.maximum(ESat - self.var.EAct, 0.0)
        X1 = np.maximum(VPD-1,-0.5)
        ##only once###

        PAR = 0.0005 * self.var.Rsds * (1-np.exp(-0.65 * LAI))

        XX = RUE-VPR*X1
        DDM = XX * PAR * AJWA

        DDM = DDM * REG * self.SHRL

        BN = BN + DDM
        return BN

    def calculateSTL(self, BN, HUI, RTF1, RTF2):
        RF = np.maximum(0.2,RTF1*(1.-HUI)+RTF2*HUI)
        RW = RF*BN
        STL = np.maximum(0.,BN-RW)

        return STL

    def calculateYield(self,HIO,HUI,STL,TBM,SWH,SWP,HIL,N_stress):
        X1 = 100 * HUI
        AJHI = HIO * X1 / (X1+np.exp(self.SCRP3_1-self.SCRP3_2*X1))

        XX = 100.* SWH / ( SWP + 1.E-5)
        F= XX / ( XX +np.exp(self.SCRP10_1-self.SCRP10_2*XX))

        HLA=np.minimum( F * np.maximum(AJHI - HIL,0.) + HIL,.9 * TBM / ( STL+1.E-10))
        YIELD = STL * HLA * N_stress
        return YIELD

    def calculateTempStress(self,c):
        RTO = (self.var.Tavg - self.var.TB[c])/(self.var.TO[c] - self.var.TB[c])
        TS = np.where((RTO < 2) & (self.var.Tavg>self.var.TB[c]),np.sin(1.5707*RTO),0)

        return TS
    def dynamic(self):

        if dateVar['newStart']:

            self.SLA0_Irr =  np.tile(globals.inZero, (len(self.var.Crops_names), 1))
            self.SLA0_nonIrr =  np.tile(globals.inZero, (len(self.var.Crops_names), 1))

            self.var.JJP_Irr = np.tile(globals.inZero, (len(self.var.Crops_names), 1))
            self.var.JJP_nonIrr = np.tile(globals.inZero, (len(self.var.Crops_names), 1))

            self.var.JJT_Irr = np.tile(globals.inZero, (len(self.var.Crops_names), 1))
            self.var.JJT_nonIrr = np.tile(globals.inZero, (len(self.var.Crops_names), 1))

        HUI_Irr_daily = self.var.HUI_Irr_daily.copy()
        HUI_nonIrr_daily = self.var.HUI_nonIrr_daily.copy()

        HUF_Irr_daily = self.var.HUF_Irr_daily.copy()
        HUF_nonIrr_daily = self.var.HUF_nonIrr_daily.copy()

        LAI_Irr_daily = self.var.LAI_Irr_daily.copy()
        LAI_nonIrr_daily = self.var.LAI_nonIrr_daily.copy()

        height_Irr_daily = self.var.height_Irr_daily.copy()
        height_nonIrr_daily = self.var.height_nonIrr_daily.copy()

        totalBm_Irr_daily = self.var.totalBm_Irr_daily.copy()
        totalBm_nonIrr_daily = self.var.totalBm_nonIrr_daily.copy()

        # Crop dormancy module
        FracDH = self.calculateDaylightHours()

        # Determine whether the crop enters dormancy.
        if dateVar['newStart']:
            self.SHRL = globals.inZero.copy() + 1
        Frac = FracDH * self.SHRL

        # In addition to natural senescence after maturity, crop dormancy
        # reduces LAI in response to (1) insufficient daylight and
        # (2) minimum temperatures below 1 degC. Dormancy behavior near the
        # equator still needs to be addressed.
        self.SHRL = np.where(self.HRLT>self.WDRM,1,0)
        FHR = np.where(self.SHRL==1,0,1-self.HRLT/self.WDRM)

        for c in range(len(self.var.Crops_names)):

            self.var.HUI_Irr_daily[c] = np.where((self.var.JJT_Irr[c] > 0)&(dateVar['doy'] != self.AJHU),\
                                                                np.maximum(self.var.Tavg - self.var.TB[c],0)/self.var.PHU_Irr[c] + HUI_Irr_daily[c],\
                                                                    np.where(self.var.JJT_Irr[c] == 2,1,0))
            self.var.HUI_nonIrr_daily[c] = np.where((self.var.JJT_nonIrr[c] > 0)&(dateVar['doy'] != self.AJHU),\
                                                                    np.maximum(self.var.Tavg - self.var.TB[c],0)/self.var.PHU_nonIrr[c] + HUI_nonIrr_daily[c],\
                                                                        np.where(self.var.JJT_nonIrr[c] == 2,1,0))

            ##crop is planted 0:not planted;
            # 1: planted but not matured;
            # 2. Matured has not been harvested.;
            # 3: harvested immediately.
            self.var.JJT_Irr[c] = np.where(self.var.JJT_Irr[c] > 0,\
                                        np.where(self.var.HUI_Irr_daily[c] >= 1,\
                                                 np.where(self.var.GrowPeriod_Irr[c] < self.var.GP_Irr[c],2,3),\
                                                    np.where(self.var.GrowPeriod_Irr[c] < self.var.GP_Irr[c]*1.2,1,3)),0)
            self.var.JJT_nonIrr[c] = np.where(self.var.JJT_nonIrr[c] > 0,\
                                            np.where(self.var.HUI_nonIrr_daily[c] >= 1,\
                                                     np.where(self.var.GrowPeriod_nonIrr[c] < self.var.GP_nonIrr[c],2,3),\
                                                        np.where(self.var.GrowPeriod_nonIrr[c] < self.var.GP_nonIrr[c]*1.2,1,3)),0)

            # Determine whether the crop has emerged.
            self.var.JJP_Irr[c] = np.where(self.var.HUI_Irr_daily[c] >= (self.var.HUG[c] / self.var.PHU_Irr[c]),1,0)
            self.var.JJP_nonIrr[c] = np.where(self.var.HUI_nonIrr_daily[c] >= (self.var.HUG[c] / self.var.PHU_nonIrr[c]),1,0)


            self.var.HUF_Irr_daily[c],self.var.dHUF_Irr_daily[c] = np.where(\
                (self.var.JJP_Irr[c] == 1) & (self.var.JJT_Irr[c] == 1),\
                    self.calculateHUF(HUI_Irr_daily[c],HUF_Irr_daily[c],c),0)
            self.var.HUF_nonIrr_daily[c],self.var.dHUF_nonIrr_daily[c] = np.where(\
                (self.var.JJP_nonIrr[c] == 1) & (self.var.JJT_nonIrr[c] == 1),\
                    self.calculateHUF(HUI_nonIrr_daily[c],HUF_nonIrr_daily[c],c),0)

            self.SLA0_Irr[c] = np.where((self.var.JJP_Irr[c] == 1) & (self.var.JJT_Irr[c] == 1)\
                                                        & (self.var.HUI_Irr_daily[c] >= self.var.HUIS[c]) &\
                                                            (self.SLA0_Irr[c] == -9999), self.var.LAI_Irr_daily[c], \
                                                                np.where(self.SLA0_Irr[c] > 0,self.SLA0_Irr[c],-9999))
            self.SLA0_nonIrr[c] = np.where((self.var.JJP_nonIrr[c] == 1) & (self.var.JJT_nonIrr[c] == 1)\
                                                        & (self.var.HUI_nonIrr_daily[c] >= self.var.HUIS[c]) &\
                                                            (self.SLA0_nonIrr[c] == -9999),self.var.LAI_nonIrr_daily[c], \
                                                                np.where(self.SLA0_nonIrr[c] > 0,self.SLA0_nonIrr[c],-9999))

            self.var.TS_daily[c] = np.where(self.var.JJP_Irr[c] == 1,self.calculateTempStress(c),1)

            self.var.REG_Irr_daily[c] = np.where(self.var.JJP_Irr[c] == 1,\
                                                                np.minimum(self.var.TS_daily[c],1),1)
            self.var.REG_nonIrr_daily[c] = np.where(self.var.JJP_nonIrr[c] == 1,\
                                                                    np.minimum(self.var.TS_daily[c],self.var.WS_nonIrr_daily[c]),1)

            LAI_Irr = np.where((self.var.JJP_Irr[c] == 1) & (self.var.JJT_Irr[c] == 1),\
                                                            self.calculatedLAI(\
                                                                self.var.HUI_Irr_daily[c],\
                                                                    self.var.dHUF_Irr_daily[c],\
                                                                        LAI_Irr_daily[c],\
                                                                            self.SLA0_Irr[c],\
                                                                                self.var.WS_Irr_daily[c],Frac,c),\
                                                                                    np.where(self.var.JJT_Irr[c] > 1,LAI_Irr_daily[c],0))
            LAI_nonIrr = np.where((self.var.JJP_nonIrr[c] == 1) & (self.var.JJT_nonIrr[c] == 1),\
                                                            self.calculatedLAI(\
                                                                self.var.HUI_nonIrr_daily[c],\
                                                                    self.var.dHUF_nonIrr_daily[c],\
                                                                        LAI_nonIrr_daily[c],\
                                                                            self.SLA0_nonIrr[c],\
                                                                                self.var.WS_nonIrr_daily[c],Frac,c),\
                                                                                    np.where(self.var.JJT_nonIrr[c] > 1,LAI_nonIrr_daily[c],0))

            AJWA_Irr = self.calculateBiomassDeclineFactor(self.var.HUI_Irr_daily[c], c)
            AJWA_nonIrr = self.calculateBiomassDeclineFactor(self.var.HUI_nonIrr_daily[c], c)


            self.var.RD_Irr_daily[c] = np.where((self.var.JJP_Irr[c] == 1) & (self.var.JJT_Irr[c] == 1),\
                                                                self.calculatedRD(self.var.HUI_Irr_daily[c],self.var.RDM[c]),np.where(self.var.JJT_Irr[c] > 1,self.var.RD_Irr_daily[c],0))
            self.var.RD_nonIrr_daily[c] = np.where((self.var.JJP_nonIrr[c] == 1) & (self.var.JJT_nonIrr[c] == 1),\
                                                                    self.calculatedRD(self.var.HUI_nonIrr_daily[c],self.var.RDM[c]),np.where(self.var.JJT_nonIrr[c] > 1,self.var.RD_nonIrr_daily[c],0))
            self.var.height_Irr_daily[c] = np.where((self.var.JJP_Irr[c] == 1) & (self.var.JJT_Irr[c] == 1),\
                                                                self.calculatedHeight(self.var.HUF_Irr_daily[c],self.var.LAI_Irr_daily[c],height_Irr_daily[c],self.var.HCM[c]),np.where(self.var.JJT_Irr[c] > 1,height_Irr_daily[c],0))
            self.var.height_nonIrr_daily[c] = np.where((self.var.JJP_nonIrr[c] == 1) & (self.var.JJT_nonIrr[c] == 1),\
                                                                    self.calculatedHeight(self.var.HUF_nonIrr_daily[c],self.var.LAI_nonIrr_daily[c],height_nonIrr_daily[c],self.var.HCM[c]),np.where(self.var.JJT_nonIrr[c] > 1,height_nonIrr_daily[c],0))


            F = np.where(self.var.TMin >-1,np.where(self.SHRL>0,0,FHR),\
            np.maximum(np.abs(self.var.TMin) / (np.abs(self.var.TMin) + np.exp(self.FDMB1[c] - self.FDMB2[c] * np.abs(self.var.TMin))),FHR))

            self.var.LAI_Irr_daily[c] = np.where((self.var.JJP_Irr[c] == 1) & (self.var.JJT_Irr[c] == 1),np.maximum(0.02,LAI_Irr * (1-F)),np.where(self.var.JJT_Irr[c] > 1,LAI_Irr_daily[c],0))
            self.var.LAI_nonIrr_daily[c] = np.where((self.var.JJP_nonIrr[c] == 1) & (self.var.JJT_nonIrr[c] == 1),np.maximum(0.02,LAI_nonIrr * (1-F)),np.where(self.var.JJT_nonIrr[c] > 1,LAI_nonIrr_daily[c],0))

            bio_dec_Irr = np.where((self.var.STLBm_Irr_daily[c]>0)&(self.var.JJT_Irr[c] == 1),F*self.var.STLBm_Irr_daily[c],0)
            bio_dec_nonIrr = np.where((self.var.STLBm_nonIrr_daily[c]>0)&(self.var.JJT_nonIrr[c] == 1),F*self.var.STLBm_nonIrr_daily[c],0)

            self.var.STLBm_Irr_daily[c] = self.var.STLBm_Irr_daily[c] - bio_dec_Irr
            self.var.STLBm_nonIrr_daily[c] = self.var.STLBm_nonIrr_daily[c] - bio_dec_nonIrr


            totalBm_Irr_daily[c] = totalBm_Irr_daily[c] - bio_dec_Irr
            totalBm_nonIrr_daily[c] = totalBm_nonIrr_daily[c] - bio_dec_nonIrr

            self.var.totalBm_Irr_daily[c] = np.where((self.var.JJP_Irr[c] == 1) & (self.var.JJT_Irr[c] == 1),\
                                                                self.calculateBioMass(\
                                                                    self.var.LAI_Irr_daily[c],\
                                                                        self.var.REG_Irr_daily[c],\
                                                                            totalBm_Irr_daily[c],\
                                                                                    self.var.VPR[c],self.RUE[c],AJWA_Irr),
                                                                    np.where(self.var.JJT_Irr[c] > 1,self.var.totalBm_Irr_daily[c],0))

            self.var.totalBm_nonIrr_daily[c] = np.where((self.var.JJP_nonIrr[c] == 1) & (self.var.JJT_nonIrr[c] == 1),\
                                                                self.calculateBioMass(\
                                                                    self.var.LAI_nonIrr_daily[c],\
                                                                        self.var.REG_nonIrr_daily[c],\
                                                                            totalBm_nonIrr_daily[c],\
                                                                                     self.var.VPR[c],self.RUE[c],AJWA_nonIrr),\
                                                                            np.where(self.var.JJT_nonIrr[c] > 1,self.var.totalBm_nonIrr_daily[c],0))
            self.var.STLBm_Irr_daily[c] = np.where((self.var.JJP_Irr[c] == 1) & (self.var.JJT_Irr[c] > 0),\
                                                                self.calculateSTL(self.var.totalBm_Irr_daily[c],\
                                                                    self.var.HUI_Irr_daily[c],self.var.RTF1[c],self.var.RTF2[c]),\
                                                                        np.where(self.var.JJT_Irr[c] > 1,self.var.STLBm_Irr_daily[c],0))
            self.var.STLBm_nonIrr_daily[c] = np.where((self.var.JJP_nonIrr[c] == 1) & (self.var.JJT_nonIrr[c] > 0),\
                                                                self.calculateSTL(self.var.totalBm_nonIrr_daily[c],\
                                                                    self.var.HUI_nonIrr_daily[c],self.var.RTF1[c],self.var.RTF2[c]),\
                                                                        np.where(self.var.JJT_nonIrr[c] > 1,self.var.STLBm_nonIrr_daily[c],0))


            self.var.actYield_Irr[c] = np.where(self.var.JJT_Irr[c] == 3,\
                                                                self.calculateYield(self.var.HIA[c],
                                                                self.var.HUI_Irr_daily[c],
                                                                self.var.STLBm_Irr_daily[c],
                                                                self.var.totalBm_Irr_daily[c],
                                                                self.var.actTransForHI_Irr_daily[c],
                                                                self.var.potTransForHI_Irr_daily[c],
                                                                self.var.HIL[c],
                                                                self.var.N_stress_Irr[c]),0)
            self.var.actYield_nonIrr[c] = np.where(self.var.JJT_nonIrr[c] == 3,\
                                                                self.calculateYield(self.var.HIA[c],
                                                                self.var.HUI_nonIrr_daily[c],
                                                                self.var.STLBm_nonIrr_daily[c],
                                                                self.var.totalBm_nonIrr_daily[c],
                                                                self.var.actTransForHI_nonIrr_daily[c],
                                                                self.var.potTransForHI_nonIrr_daily[c],
                                                                self.var.HIL[c],
                                                                self.var.N_stress_nonIrr[c]),0)

            vars(self.var)['actYield_Irr'+str(c)] = self.var.actYield_Irr[c]
            vars(self.var)['actYield_nonIrr'+str(c)] = self.var.actYield_nonIrr[c]

    def scrp_to_xy(self,scrp_point):
        """
        Decode an SCRP-formatted value (for example, 15.01) into
        its X and Y components (X=15, Y=0.01).
        """
        X = int(scrp_point)  # Use the integer part as X.
        Y = round(scrp_point - int(scrp_point), 2)  # Use the fractional part as Y.
        return X, Y

    def calculate_s_curve_params(self,X1,Y1,X2,Y2):
        """
        Calculate B1 and B2 from two decoded SCRP control points.

        :param X1: X-coordinate of the first control point.
        :param Y1: Y-coordinate of the first control point.
        :param X2: X-coordinate of the second control point.
        :param Y2: Y-coordinate of the second control point.
        :return: B1 and B2 parameters of the S-curve.
        """

        xx = np.log(X1 /Y1-X1)
        B2 = (xx - np.log(X2 / Y2 - X2)) / (X2 - X1)
        B1 = xx + X1 * B2

        return np.round(B1, 6), np.round(B2, 6)

    def f(self,g1, x4, x5):
        z1 = np.exp(x4 - x5 * g1)
        denominator = g1 + z1
        if np.abs(denominator) < 1e-15:
            return np.inf
        return g1 / denominator - 0.9

    def fprime(self,g1, x4, x5):
        z1 = np.exp(x4 - x5 * g1)
        denominator = g1 + z1
        if np.abs(denominator) < 1e-15:
            return np.inf
        return z1 * (1.0 + x5 * g1) / (denominator**2)
