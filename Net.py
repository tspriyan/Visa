from random import *
from units import *
import xlwt
import matplotlib.pyplot as plt

# the ViSA neural network
class Net():
    # simulation parameters
    seed = 0
    dT = 0.1

    # stimulation params
    cueTime = 0

    # Sequential presentation settings; RSVP = 0 for simultaneous presentation
    RSVP = 1
    lag = 1
    T2asLast = 0

    # Simultaneous presentation settings (loaded automatically if RSVP = 0)
    numStim = 4
    numTarget = 4
    numHighP = 4
    
    # if mask = 1, half of the stimuli becomes mask for the other half (if even numStim, add an object)
    mask = 1
    
    # data to save
    saveData = 1
    saveK = 1
    saveCueTime = 0

    # Trail parameters
    numOsc = 10
    numInt = 1
    numMod = 4
    
    durata = 1500/dT 
    sduratra = 100
    
    pos = [1,1,1,1,1]
    
    cuetime = 0
            
    def __init__ (self, seed, nObj, param):
        # define the network parameters
        self.param = param

        # starting seed for the network
        self.seed = seed

        # number of objects
        self.numObj = nObj


        # Activity
        self.ActOscLP = []
        self.ActOscHP = []
        self.ActOscGW = []
        self.ActOscVSWM = []
            
        self.ActIntLP = []
        self.ActIntHP = []
        self.ActIntGW = []
        self.ActIntVSWM = []
        
        for i in range (self.numObj):
            self.ActOscLP.append(0)
            self.ActOscHP.append(0)
            self.ActOscGW.append(0)
            self.ActOscVSWM.append(0)
            
            self.ActIntLP.append(0)
            self.ActIntHP.append(0)
            self.ActIntGW.append(0)
            self.ActIntVSWM.append(0)
        
        self.MeanLPact = [[],[],[],[],[]]
        self.MeanHPact = [[],[],[],[],[]]
        self.MeanGWact = [[],[],[],[],[]]
        self.MeanVSWMact = [[],[],[],[],[]]
            
        self.MeanLPOsc = [[],[],[],[],[]]
        self.MeanHPOsc = [[],[],[],[],[]]
        self.MeanGWOsc = [[],[],[],[],[]]
        self.MeanVSWMOsc = [[],[],[],[],[]]

        self.MeanLPInt = [[],[],[],[],[]]
        self.MeanHPInt = [[],[],[],[],[]]
        self.MeanGWInt = [[],[],[],[],[]]
        self.MeanVSWMInt = [[],[],[],[],[]]
          
        #target stim
        self.tstream = [100,0,0,0,0]
        self.tstim = [2000,0,0,0,0]
        self.dist = [0,1,1,1 ,1]
        
        # selection efficiency params
        highEff = 1
        lowEff = 0

        if highEff==1:
            self.attnbiastarg= gauss(0.1,0.5)
            self.attnbiasdist= gauss(0.02,0.5)
    
        if lowEff==1:
            self.attnbiastarg= gauss(0.1,0.5)
            self.attnbiasdist= gauss(0.02,0.5)
        
        # network units (list)
        self.units = []

        # network connections (list of lists)
        self.conn = []

        # area and layer unit lists
        self.LPmapOsc = []
        self.HPmapOsc = []
        self.GWOsc = []
        self.VSWMOsc = []

        self.LPmapInt = []
        self.HPmapInt = []
        self.GWInt = []
        self.VSWMInt = []

        self.SpikingUnits = []
        self.FeedingUnits = []
        self.GatingUnits = []
        self.BlockingUnits = []

        self.InputLayer = []
        self.MaskingLayer = []
        self.GWInterneuron = []
        self.VSWMInterneuron = []
        

        for n in range(self.numObj):
            self.createStream()            
            self.createConnStream(n)
            self.createConnGate(n)
            self.createConnStreamBU(n)         
            self.createConnStreamTD(n)   
        
        self.createConnInibStream()

        if self.mask==1 and self.numObj>1:
            self.createConnMask()
            
    # create units and connections within a stream (it includes all the units coding for an object)
    def createStream(self):
        items = [] # useful to define group of oscillators
        
        # first create the input unit (check if the "units" list is void)
        try:
            self.units.append(Inputunit(self.units[-1].id + 1)) 
        except:
            self.units.append(Inputunit(0))
            
        self.InputLayer.append(self.units[-1].id)

        # create the LPmap fast and slow excitatory units + masking units
        for o in range(Net.numOsc):
            self.units.append(Oscillator(self.units[-1].id+1))
            items.append(self.units[-1].id)
            
        self.LPmapOsc.append(items)
        
        items = []

        self.units.append(Integrator(self.units[-1].id+1, "LPmap"))
        self.LPmapInt.append(self.units[-1].id)

        self.units.append(MaskingUnit(self.units[-1].id+1))
        self.MaskingLayer.append(self.units[-1].id)

        # create the HPmap fast and slow units
        for o in range(Net.numOsc):
            self.units.append(Oscillator(self.units[-1].id+1))
            items.append(self.units[-1].id)

        self.HPmapOsc.append(items)
        items = []
        
        self.units.append(Integrator(self.units[-1].id+1, "HPmap"))
        self.HPmapInt.append(self.units[-1].id)

        # create the GW fast and slow units + inh interneurons + gating units
        for o in range(Net.numOsc):
            self.units.append(Oscillator(self.units[-1].id+1))
            items.append(self.units[-1].id)

        self.GWOsc.append(items)
        items = []
        
        self.units.append(Integrator(self.units[-1].id+1, "GW"))
        self.GWInt.append(self.units[-1].id)
        self.units.append(Interneuron(self.units[-1].id+1))
        self.GWInterneuron.append(self.units[-1].id)
        self.units.append(SpikingUnit(self.units[-1].id+1))
        self.SpikingUnits.append(self.units[-1].id)
        self.units.append(FeedingUnit(self.units[-1].id+1))
        self.FeedingUnits.append(self.units[-1].id)
        self.units.append(GateUnit(self.units[-1].id+1))
        self.GatingUnits.append(self.units[-1].id)
        self.units.append(BlockingUnit(self.units[-1].id+1))
        self.BlockingUnits.append(self.units[-1].id)

        # create the VSWM fast and slow units + inh interneurons
        for o in range(Net.numOsc):
            self.units.append(Oscillator(self.units[-1].id+1))
            items.append(self.units[-1].id)

        self.VSWMOsc.append(items)
        items = []

        self.units.append(Integrator(self.units[-1].id+1, "VSWM"))
        self.VSWMInt.append(self.units[-1].id)

        self.units.append(Interneuron(self.units[-1].id+1))
        self.VSWMInterneuron.append(self.units[-1].id)
        
    # create the Osc-Int connections inside a stream
    def createConnStream(self, n):
        # create Osc->Int connections in LPmap
        
        for i in self.LPmapOsc[n]:
            self.conn.append(Connection(True, self.units[self.LPmapInt[n]], self.units[i], self.param["gOsc_IntLp"], self.param["delZero"], self.param["nOsc_Int"], self.param["QOsc_Int"]))

        # create Osc->Int connections in HPmap
        for i in self.HPmapOsc[n]:
            self.conn.append(Connection(True, self.units[self.HPmapInt[n]], self.units[i], self.param["gOsc_IntHp"], self.param["delZero"], self.param["nOsc_Int"], self.param["QOsc_Int"]))

        # create Osc->Int connections in GW
        for i in self.GWOsc[n]:
            self.conn.append(Connection(True, self.units[self.GWInt[n]], self.units[i], self.param["gOsc_IntGW"], self.param["delZero"], self.param["nOsc_Int"], self.param["QOsc_Int"]))

        # create Osc->Int connections in VSWM
        for i in self.VSWMOsc[n]:
            self.conn.append(Connection(True, self.units[self.VSWMInt[n]], self.units[i], self.param["gOsc_IntVSWM"], self.param["delZero"], self.param["nOsc_Int"], self.param["QOsc_Int"]))

        # create Int->Osc connections in VSWM
        for i in self.VSWMOsc[n]:
            self.conn.append(Connection(True, self.units[i], self.units[self.VSWMInt[n]], self.param["gInt_Osc"], self.param["delZero"], self.param["nOscExc"], self.param["QOscExc"]))

    def createConnGate(self, n):
        # create connections between units coding for the same object in the gate layer
        self.conn.append(Connection(True, self.units[self.FeedingUnits[n]], self.units[self.SpikingUnits[n]], self.param["gSp_Feed"], self.param["delZero"], self.param["nToFeed"], self.param["QToFeed"]))
        self.conn.append(Connection(True, self.units[self.BlockingUnits[n]], self.units[self.SpikingUnits[n]], self.param["gSp_Block"], self.param["delZero"], self.param["nToFeed"], self.param["QToFeed"]))
        self.conn.append(Connection(False, self.units[self.SpikingUnits[n]], self.units[self.BlockingUnits[n]], self.param["gBlock_Sp"], self.param["delZero"], self.param["nSp_Sp"], self.param["QSp_Sp"]))
        self.conn.append(Connection(False, self.units[self.GatingUnits[n]], self.units[self.BlockingUnits[n]], self.param["gBlock_Gate"], self.param["delZero"], self.param["nSp_Sp"], self.param["QSp_Sp"]))
        self.conn.append(Connection(False, self.units[self.SpikingUnits[n]], self.units[self.GatingUnits[n]], self.param["gGate_Sp"], self.param["delGate_Sp"], self.param["nInt_Sp"], self.param["QInt_Sp"]))
        # create connections from the Gating Unit and the corresponding oscillators in the GW module
        for i in self.GWOsc[n]:
            self.conn.append(Connection(False, self.units[i], self.units[self.GatingUnits[n]], self.param["gGate_Osc"], self.param["delGate_Osc"], self.param["nGate_Osc"], self.param["QGate_Osc"]))

    def createConnStreamBU(self, n):      
        uA = []
        uE = []
        sxitm=[]
        import random 
        # first connect input unit to scillators in the LPmap
        uA = self.LPmapOsc[n]
        for i in uA:
            self.conn.append(Connection(True, self.units[i], self.units[self.InputLayer[n]], self.param["gBUlp"], self.param["delBU"], self.param["nOscExc"], self.param["QOscExc"]))
            
        # LPmap->Hpmap
        uA = self.HPmapOsc[n]
        uE = self.LPmapOsc[n]
        for i in uE:
            sxitm=random.sample(uA, 6)
            for j in sxitm:
                self.conn.append(Connection(True, self.units[j], self.units[i], self.param["gBUhp"], self.param["delBU"], self.param["nOscExc"], self.param["QOscExc"]))
                                    
        # HPmap->GW
        uA = self.GWOsc[n]
        uE = self.HPmapOsc[n]
        for i in uE:
            sxitm=random.sample(uA, 6)
            for j in sxitm:
                self.conn.append(Connection(True, self.units[j], self.units[i], self.param["gBUgw"], self.param["delBU"], self.param["nOscExc"], self.param["QOscExc"]))

        # GW->VSWM
        uA = self.VSWMOsc[n]
        uE = self.GWOsc[n]
        for i in uE:
            sxitm=random.sample(uA, 6)
            for j in sxitm:
                self.conn.append(Connection(True, self.units[j], self.units[i], self.param["gBUwm"], self.param["delBU"], self.param["nOscExc"], self.param["QOscExc"]))

        # GW->VSWM (integrator-oscillators)
        uA = self.VSWMOsc[n]
        
        for i in self.VSWMOsc[n]:
            self.conn.append(Connection(True, self.units[i], self.units[self.GWInt[n]], self.param["gBUwmInt"], self.param["delBU"], self.param["nBUpGW_WM"], self.param["QBUpGW_WM"]))
 
        # GW integrator -> Spiking Unit (to activate the gating dynamics)
        self.conn.append(Connection(True, self.units[self.SpikingUnits[n]], self.units[self.GWInt[n]], self.param["gGWint_Sp"], self.param["delBU"]/2, self.param["nGW_Sp"], self.param["QGW_Sp"]))         

    def createConnStreamTD(self, n):
        # Top-down connections are established between 1 efferent integrator and the 10 afferent oscillators
        # HPmap->LPmap
        for i in self.LPmapOsc[n]:
            self.conn.append(Connection(True, self.units[i], self.units[self.HPmapInt[n]], self.param["gTDlp"], self.param["delTD"]+self.param["delTDincr"], self.param["nOscExc"], self.param["QOscExc"]))

        # GW->HPmap
        for i in self.HPmapOsc[n]:
            self.conn.append(Connection(True, self.units[i], self.units[self.GWInt[n]], self.param["gTDhp"], self.param["delTD"]+self.param["delTDincr"], self.param["nOscExc"], self.param["QOscExc"]))
            
        # VSWM->HPmap
        for i in self.HPmapOsc[n]:
            self.conn.append(Connection(True, self.units[i], self.units[self.VSWMInt[n]], self.param["gTDlp"]/10, self.param["delTD"]+self.param["delTDincr"]*3, self.param["nOscExc"], self.param["QOscExc"]))

#         VSWM->GW
        for i in self.GWOsc[n]:
            self.conn.append(Connection(True, self.units[i], self.units[self.VSWMInt[n]], self.param["gTDgw"]/10, self.param["delTD"]+self.param["delTDincr"], self.param["nOscExc"], self.param["QOscExc"]))

        # Inhibitory TD connections VSWM->FeedingUnits
        self.conn.append(Connection(False, self.units[self.FeedingUnits[n]], self.units[self.VSWMInt[n]], self.param["gWMint_Feed"], self.param["delTD"]+self.param["delTDincr"]*2, self.param["nWM_Int"], self.param["QWM_Int"]))

    def createConnInibStream(self):
        # create excitatory connections Osc->Inhibitory Interneuron in GW
        for n in range(self.numObj):
            for i in self.GWOsc[n]:
                self.conn.append(Connection(True, self.units[self.GWInterneuron[n]], self.units[i], self.param["gOsc_Inh"], self.param["delOsc_Inh"], self.param["nOsc_Inh"], self.param["QOsc_Inh"]))
        
        #  create inhibitory connections Interneuron->Osc in GW
        for n in range(self.numObj):
            for m in range (self.numObj):
                if m!=n:
                    uA=self.GWOsc[n]
                    uE=self.GWInterneuron[m]
                    for i in uA:
                        self.conn.append(Connection(False, self.units[i], self.units[uE], self.param["gINHgw"], self.param["delInh_Osc"], self.param["nInh_Osc"], self.param["QInh_Osc"]))

        # create excitatory connections Osc->Inhibitory Interneuron in VSWM
        for n in range(self.numObj):            
            for i in self.VSWMOsc[n]:
                self.conn.append(Connection(True, self.units[self.VSWMInterneuron[n]], self.units[i], self.param["gOsc_Inh"], self.param["delOsc_Inh"], self.param["nOsc_Inh"], self.param["QOsc_Inh"]))
        
        #  create inhibitory connections Interneuron->Osc in VSWM
        for n in range(self.numObj):
            for m in range (self.numObj):
                if m!=n:
                    uA=self.VSWMOsc[n]
                    uE=self.VSWMInterneuron[m]
                    for i in uA:
                        self.conn.append(Connection(False, self.units[i], self.units[uE], self.param["gINHwm"], self.param["delInh_Osc"], self.param["nInh_Osc"], self.param["QInh_Osc"]))                
         
        # create inhibitory connections GateUnits[n]->GateUnits[m] for gating competition
        for m in range (self.numObj):
            for n in range (self.numObj):
                if n!=m:
                    self.conn.append(Connection(False, self.units[self.SpikingUnits[m]], self.units[self.SpikingUnits[n]], self.param["gSp_SpInter"], self.param["delSp_SpInter"], self.param["nSp_Sp"], self.param["QSp_Sp"]))
                    self.conn.append(Connection(False, self.units[self.FeedingUnits[m]], self.units[self.SpikingUnits[n]], self.param["gSp_FeedInter"], self.param["delSp_FeedInter"], self.param["nSp_Feed"], self.param["QSp_Feed"]))
                    self.conn.append(Connection(False, self.units[self.GatingUnits[m]], self.units[self.SpikingUnits[n]], self.param["gSp_GateInter"], self.param["delSp_GateInter"], self.param["nSp_Feed"], self.param["QSp_Feed"]))
                    self.conn.append(Connection(True, self.units[self.GatingUnits[m]], self.units[self.FeedingUnits[n]], self.param["gFeed_GateInter"], self.param["delFeed_GateInter"], self.param["nToGate"], self.param["QToGate"]))
                  
    def createConnMask(self):
        
        # create connections from Input Unit and Osc in LPmap to the correspondin Masking Unit
        for m in range(self.numObj):
            self.conn.append(Connection(True, self.units[self.MaskingLayer[m]], self.units[self.InputLayer[m]], self.param["gInput_Mask"]/10, self.param["delInput_Mask"], self.param["nOsc_Mask"], self.param["QOsc_Mask"]))
            for n in range(self.numObj):
                if n!=m:
                    if self.pos[n]==self.pos[m]:
                        uA= self.LPmapOsc[m]
                        for j in uA:
                            self.conn.append(Connection(False, self.units[j], self.units[self.MaskingLayer[n]], self.param["gMask_Osc"], self.param["delMask_Osc"], self.param["nMask_Osc"], self.param["QMask_Osc"]))

#    def createConnInterneuron(self):
#        # create excitatory connections Osc->Inhibitory Interneuron in GW and VSWM
#        for n in range(self.numObj):
#            for i in self.GWOsc[n]:
#                self.conn.append(Connection(True, self.units[self.GWInterneuron[n]], self.units[i], self.param["gOsc_Inh"], self.param["delOsc_Inh"], self.param["nOsc_Inh"], self.param["QOsc_Inh"]))
#        for n in range(self.numObj):            
#            for i in self.VSWMOsc[n]:
#                self.conn.append(Connection(True, self.units[self.VSWMInterneuron[n]], self.units[i], self.param["gOsc_Inh"], self.param["delOsc_Inh"], self.param["nOsc_Inh"], self.param["QOsc_Inh"]))

#    def createConnOscInh(self, n, m):
#        # create inhibitory connections Interneuron->Osc in GW and VSWM
#        for i in GWOsc[m]:
#            self.conn.append(Connection(False, self.units[i], self.units[GWInterneuron[n]], self.param["gINHgw"], self.param["delInh_Osc"], self.param["nInh_Osc"], self.param["QInh_Osc"]))
#
#        for i in VSWMOsc[m]:
#            self.conn.append(Connection(False, self.units[i], self.units[VSWMInterneuron[n]], self.param["gINHwm"], self.param["delInh_Osc"], self.param["nInh_Osc"], self.param["QInh_Osc"]))

#    def createConnGateInh(self, n, m):
#        # create inhibitory connections GateUnits[n]->GateUnits[m] for gating competition
#        self.conn.append(Connection(False, self.units[SpikingUnits[m]], self.units[SpikingUnits[n]], self.param["gSp_SpInter"], self.param["delSp_SpInter"], self.param["nSp_Sp"], self.param["QSp_Sp"]))
#        self.conn.append(Connection(False, self.units[FeedingUnits[m]], self.units[SpikingUnits[n]], self.param["gSp_FeedInter"], self.param["delSp_FeedInter"], self.param["nSp_Feed"], self.param["QSp_Feed"]))
#        self.conn.append(Connection(False, self.units[GatingUnits[m]], self.units[SpikingUnits[n]], self.param["gSp_GateInter"], self.param["delSp_GateInter"], self.param["nSp_Feed"], self.param["QSp_Feed"]))
#
#        self.conn.append(Connection(True, self.units[GatingUnits[n]], self.units[FeedingUnits[n]], self.param["gFeed_GateInter"], self.param["delFeed_GateInter"], self.param["nToGate"], self.param["QToGate"]))


    def run(self, steps):
        nStep = steps
        print (nStep)
        if nStep>0:
            for i in self.conn:
                Connection.updateAct(i,nStep)
                
        for i in range(self.numObj):
            if self.numObj>1:
                MaskingUnit.updateAct(self.units[self.MaskingLayer[i]])
                Inputunit.updateAct(self.units[self.InputLayer[i]],i,nStep,self.tstream[i],self.tstim[i])
                SpikingUnit.updateAct(self.units[self.SpikingUnits[i]])
                FeedingUnit.updateAct(self.units[self.FeedingUnits[i]])
                GateUnit.updateAct(self.units[self.GatingUnits[i]])
                BlockingUnit.updateAct(self.units[self.BlockingUnits[i]])
                
            for j in self.LPmapOsc[i]:
                Oscillator.updateAct(self.units[j],i,nStep,self.HPmapOsc[i],self.dist,self.attnbiastarg,self.attnbiasdist)
            for j in self.HPmapOsc[i]:
                Oscillator.updateAct(self.units[j],i,nStep,self.HPmapOsc[i],self.dist,self.attnbiastarg,self.attnbiasdist)
            for j in self.GWOsc[i]:
                Oscillator.updateAct(self.units[j],i,nStep,self.HPmapOsc[i],self.dist,self.attnbiastarg,self.attnbiasdist)
            for j in self.VSWMOsc[i]:
                Oscillator.updateAct(self.units[j],i,nStep,self.HPmapOsc[i],self.dist,self.attnbiastarg,self.attnbiasdist)
           
            Integrator.updateAct(self.units[self.LPmapInt[i]])
            Integrator.updateAct(self.units[self.HPmapInt[i]])
            Integrator.updateAct(self.units[self.GWInt[i]])
            Integrator.updateAct(self.units[self.VSWMInt[i]])
            
            Interneuron.updateAct(self.units[self.GWInterneuron[i]])
            Interneuron.updateAct(self.units[self.VSWMInterneuron[i]])
        
            if nStep%10==0:
        
                self.MeanLPact[i].append(((self.ActOscLP[i]/self.numOsc)+(self.ActIntLP[i]))/2)
                self.MeanHPact[i].append(((self.ActOscHP[i]/self.numOsc)+(self.ActIntHP[i]))/2)
                self.MeanGWact[i].append(((self.ActOscGW[i]/self.numOsc)+(self.ActIntGW[i]))/2)
                self.MeanVSWMact[i].append(((self.ActOscVSWM[i]/self.numOsc)+(self.ActIntVSWM[i]))/2)
                        
                self.MeanLPOsc[i].append(self.ActOscLP[i]/self.numOsc)
                self.MeanHPOsc[i].append(self.ActOscHP[i]/self.numOsc)
                self.MeanGWOsc[i].append(self.ActOscGW[i]/self.numOsc)
                self.MeanVSWMOsc[i].append(self.ActOscVSWM[i]/self.numOsc)
            
                self.MeanLPInt[i].append(self.ActIntLP[i])
                self.MeanHPInt[i].append(self.ActIntHP[i])
                self.MeanGWInt[i].append(self.ActIntGW[i])
                self.MeanVSWMInt[i].append(self.ActIntVSWM[i])
            
                self.ActOscLP[i]=0
                self.ActOscHP[i]=0
                self.ActOscGW[i]=0
                self.ActOscVSWM[i]=0
            
                self.ActIntLP[i]=0
                self.ActIntHP[i]=0
                self.ActIntGW[i]=0
                self.ActIntVSWM[i]=0
            
            else:
                for j in self.LPmapOsc[i]:
                    self.ActOscLP[i]=self.ActOscLP[i]+self.units[j].act[nStep]
            
                for j in self.HPmapOsc[i]:
                    self.ActOscHP[i]=self.ActOscHP[i]+self.units[j].act[nStep]
                    
                for j in self.GWOsc[i]:
                    self.ActOscGW[i]=self.ActOscGW[i]+self.units[j].act[nStep]

                for j in self.VSWMOsc[i]:
                    self.ActOscVSWM[i]=self.ActOscVSWM[i]+self.units[j].act[nStep]

                self.ActIntLP[i]=self.ActIntLP[i]+self.units[self.LPmapInt[i]].act[nStep]
                self.ActIntHP[i]=self.ActIntHP[i]+self.units[self.HPmapInt[i]].act[nStep]
                self.ActIntGW[i]=self.ActIntGW[i]+self.units[self.GWInt[i]].act[nStep]
                self.ActIntVSWM[i]=self.ActIntVSWM[i]+self.units[self.VSWMInt[i]].act[nStep]
                
    
    def save(self, steps):
        wb = xlwt.Workbook()
        ws = wb.add_sheet('TestSheet')

        for t in range(1000):
            for u in self.units:
                ws.write(t, u.id, u.act[t])

        wb.save('example.xls')

# network parameters
param = {
    # network parameters
    'numMod' : 4,
    'numOsc' : 10,
    # inter-stream delay parameters
    'delBU' : 5,
    'delTD' : 8,
    'delTDincr' : 3,
    # intra-module connection parameters
    'delZero' : 0.1,
    'gInt_Osc' : 0.5,
    'gOsc_IntHp' : 0.02,
    'gOsc_IntLp' : 0.01,
    'gOsc_IntGW' : 0.006,
    'gOsc_IntVSWM' : 0.01,
    'nOsc_Int' : 50,
    'QOsc_Int': 0.75,
       
     # inter-stream connection strengths
    'gBUlp' : 0.6,
    'gBUhp' : 0.2,
    'gBUgw' : 0.02,
    'gBUwm' : 0.02,
    'gBUwmInt' : 0.1,
    'gTDlp' : 0.16,
    'gTDhp' : 0.3,
    'gTDgw' : 0.1,

    'gINHgw' : 0.01,
    'gINHwm' : 0.06,
    
     # n and Q parameters for different kind of connections
    'nOscExc' : 4,
    'QOscExc' : 0.7,
    'ntdGW_HP' : 6,
    'QtdGW_HP' : 0.3,
       
     # inh interneurons
    'nOsc_Inh' : 10,
    'QOsc_Inh' : 0.5,
    'nInh_Osc' : 2,
    'QInh_Osc' : 0.5,
    'gOsc_Inh' : 0.1,
    'delOsc_Inh' : 1,
    'delInh_Osc' : 2,
       
    # integrators
    'nBUpGW_WM' : 50,
    'QBUpGW_WM' : 1.2,
       
    # masking unit
    'nOsc_Mask' : 20,
    'QOsc_Mask' : 0.7,
    'nMask_Osc' : 50,
    'QMask_Osc' : 0.85,
    'gInput_Mask' : 1,
    'delInput_Mask' : 1,
    'gMask_Osc' : 7,
    'delMask_Osc' : 1,
       
    # intra-stream gating unit connection parameters
    'nToFeed' : 4,
    'QToFeed' : 0.7,
    'nToGate' : 4,
    'QToGate' : 3,
    'gSp_Feed' : 1,
    'gSp_Block' : 1,
    'gBlock_Sp' : 1.5,
    'gBlock_Gate' : 1.5,
    'gGate_Sp' : 0.6,
    'delGate_Sp' : 2,
    'nGate_Osc' : 100,
    'QGate_Osc' : 0.45,
    'gGate_Osc' : 0.15,
    'delGate_Osc' : 1,
    'nSp_Sp' : 20,
    'QSp_Sp' : 0.2,
    'nSp_Feed' : 50,
    'QSp_Feed' : 0.5,
    'nGW_Sp' : 20,
    'QGW_Sp' : 0.3,
    'nWM_Int' : 50,
    'QWM_Int' : 1,
    'nInt_Sp' : 100,
    'QInt_Sp' : 0.45,
    'gGWint_Sp' : 0.3,
    'gWMint_Feed' : 5,
    
     # inter-stream gating unit connection parameters
    'gSp_SpInter' : 1,
    'delSp_SpInter' : 2,
    'gSp_FeedInter' : 5,
    'delSp_FeedInter' : 2,
    'gSp_GateInter' : 1,
    'delSp_GateInter' : 1,
    # fix connection to gating unit
    'gFeed_GateInter' : 1,
    'delFeed_GateInter' : 2
    }

trials=1
for tr in range(trials):
    lag1=Net(0, 5, param)
    for i in range(int(Net.durata-1)):
        lag1.run(i)
    
#PLotting]
pltcount=1
for i in range(self.numObj):
    print(i)    
    plt.figure(pltcount)
    plt.plot(self.MeanLPact[i])
    pltcount+=1
    plt.figure(pltcount)
    plt.plot(self.MeanHPact[i])
    pltcount+=1
    plt.figure(pltcount)
    plt.plot(self.MeanGWact[i])
    pltcount+=1
    plt.figure(pltcount)
    plt.plot(self.MeanVSWMact[i])
    pltcount+=1

#lag1.run(50)
#lag1.save(50)

