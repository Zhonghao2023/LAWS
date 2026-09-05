# -------------------------------------------------------------------------
# Name:       LAWS Model Initial
# Purpose:
#
# Author:      PB
#
# Created:     16/05/2016
# Copyright:   (c) PB 2016
# -------------------------------------------------------------------------
# water quality test change
from laws.hydrological_modules.miscInitial import miscInitial
from laws.hydrological_modules.initcondition import initcondition

from laws.hydrological_modules.readmeteo import readmeteo
from laws.hydrological_modules.evaporationPot import evaporationPot
from laws.hydrological_modules.inflow import inflow
from laws.hydrological_modules.snow_frost import snow_frost
from laws.hydrological_modules.soil import soil
from laws.hydrological_modules.soil_crops import soil_crops
from laws.hydrological_modules.landcoverType import landcoverType
from laws.hydrological_modules.sealed_water import sealed_water
from laws.hydrological_modules.evaporation import evaporation
from laws.hydrological_modules.evaporation_crops import evaporation_crops
from laws.hydrological_modules.groundwater import groundwater
from laws.hydrological_modules.groundwater_modflow.transient import groundwater_modflow
from laws.hydrological_modules.water_demand.water_demand import water_demand
from laws.hydrological_modules.water_demand.wastewater import waterdemand_wastewater as wastewater
from laws.hydrological_modules.capillarRise import capillarRise
from laws.hydrological_modules.interception import interception
from laws.hydrological_modules.runoff_concentration import runoff_concentration
from laws.hydrological_modules.lakes_res_small import lakes_res_small
from laws.hydrological_modules.waterbalance import waterbalance
from laws.hydrological_modules.environflow import environflow
from laws.hydrological_modules.routing_reservoirs.routing_kinematic import routing_kinematic
from laws.hydrological_modules.lakes_reservoirs import lakes_reservoirs
from laws.hydrological_modules.waterquality1 import waterquality1
from laws.agricultural_modules.BioMassCalc import BioMassCalc
from laws.agricultural_modules.new_PM_ET import new_PM_ET
from laws.agricultural_modules.crop_WU import crop_WU

from laws.management_modules.output import *
from laws.management_modules.data_handling import *
import os, glob


class Variables:
    def load_initial(self, name, default=0.0, number=None):
        """
        First it is checked if the initial value is given in the settings file

        * if it is <> None it is used directly
        * if None it is loaded from the init netcdf file

        :param name: Name of the init value
        :param default: default value -> default is 0.0
        :param number: in case of snow or runoff concentration several layers are included: number = no of the layer
        :return: spatial map or value of initial condition
        """

        if number is not None:
            name = name + str(number)

        if self.loadInit:
            map = readnetcdfInitial(self.initLoadFile, name)
            if Flags['calib']:
                self.initmap[name] = map
            return map
        else:
            return default

class Config:
    pass


class LAWSModel_ini(DynamicModel):

    """
    LAWS initialization component.
    This component initializes the model variables.
    It will call the initial part of the hydrological modules
    **Global variables**

    =====================================  ======================================================================  =====
    Variable [self.var]                    Description                                                             Unit 
    =====================================  ======================================================================  =====
    modflow                                Flag: True if modflow_coupling = True in settings file                  --   
    =====================================  ======================================================================  =====

    **Functions**
    """

    def __init__(self):
        """
        Init part of the initial part
        defines the mask map and the outlet points
        initialization of the hydrological modules
        """

        DynamicModel.__init__(self)

        self.var = Variables()
        self.conf = Config()

        # ----------------------------------------
        # include output of tss and maps
        self.output_module = outputTssMap(self)

        # include all the hydrological modules
        self.misc_module = miscInitial(self)
        self.init_module = initcondition(self)
        self.waterbalance_module = waterbalance(self)
        self.readmeteo_module = readmeteo(self)
        self.environflow_module = environflow(self)
        self.evaporationPot_module = evaporationPot(self)
        self.inflow_module = inflow(self)
        self.snowfrost_module = snow_frost(self)
        self.landcoverType_module = landcoverType(self)
        self.groundwater_module = groundwater(self)
        self.groundwater_modflow_module = groundwater_modflow(self)
        self.waterdemand_module = water_demand(self)
        self.wastewater_module = wastewater(self)
        self.capillarRise_module = capillarRise(self)
        self.interception_module = interception(self)
        self.sealed_water_module = sealed_water(self)
        self.runoff_concentration_module = runoff_concentration(self)
        self.lakes_res_small_module = lakes_res_small(self)
        self.routing_kinematic_module = routing_kinematic(self)
        self.lakes_reservoirs_module = lakes_reservoirs(self)
        self.waterquality1 = waterquality1(self)
        self.waterbalance = waterbalance(self)

        # ----------------------------------------

        # reading of the metainformation of variables to put into output netcdfs
        metaNetCDF()

        # test if ModFlow coupling is used as defined in settings file
        self.var.modflow = False
        if "modflow_coupling" in option:
            self.var.modflow = checkOption('modflow_coupling')
        self.var.crop_coupling = False
        if "crop_coupling" in option:
            self.var.crop_coupling = checkOption("crop_coupling")

        # Select exactly one soil implementation for the entire run.
        # landcoverType.dynamic() delegates soil-water calculations to this
        # object, so both initialization and dynamics follow the same switch.
        if self.var.crop_coupling:
            self.soil_module = soil_crops(self)
        else:
            self.soil_module = soil(self)

        # Select exactly one evaporation implementation for the entire run.
        # landcoverType.dynamic() delegates every land-cover calculation to
        # this object, so the dynamic path follows the same coupling choice.
        if self.var.crop_coupling:
            self.evaporation_module = evaporation_crops(self)
        else:
            self.evaporation_module = evaporation(self)

        ## MakMap: the maskmap is flexible e.g. col,row,x1,y1  or x1,x2,y1,y2
        # set the maskmap
        self.MaskMap = loadsetclone(self, 'MaskMap')
        # run intial misc to get all global variables
        self.misc_module.initial()
        self.init_module.initial()

        if self.var.crop_coupling:
            # The inherited flag controls crop bookkeeping inside CWatM
            # modules. LAWS exposes only crop_coupling as its public switch.
            self.var.includeCrops = True
        elif self.var.includeCrops:
            raise CWATMError(
                "LAWS crop processes are enabled with crop_coupling; "
                "includeCrops cannot enable them independently"
            )

        self.readmeteo_module.initial()
        self.inflow_module.initial()

        self.evaporationPot_module.initial()

        self.snowfrost_module.initial()
        self.soil_module.initial()

        # LAWS crop evapotranspiration module
        if self.var.crop_coupling:
            self.new_PM_ET_module = new_PM_ET(self)
            self.new_PM_ET_module.initial()
        # groundwater before meteo, bc it checks steady state
        if self.var.modflow and not(Flags['calib']):
            self.groundwater_modflow_module.initial()
        else:
            self.groundwater_module.initial()

        self.landcoverType_module.initial()
        self.evaporation_module.initial()
        if self.var.crop_coupling:
            self.crop_WU_module = crop_WU(self)
            self.crop_WU_module.initial()

            self.BioMassCalc_module = BioMassCalc(self)
            self.BioMassCalc_module.initial()

        self.runoff_concentration_module.initial()
        self.lakes_res_small_module.initial()

        self.routing_kinematic_module.initial()
        if checkOption('includeWaterBodies'):
            self.lakes_reservoirs_module.initWaterbodies()
            self.lakes_reservoirs_module.initial_lakes()
            self.lakes_reservoirs_module.initial_reservoirs()

        self.waterdemand_module.initial()
        self.waterbalance_module.initial()
        # calculate initial amount of water in the catchment

        self.output_module.initial()
        self.environflow_module.initial()
        self.waterquality1.initial()
