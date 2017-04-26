from __future__ import division
from random import *
# Each connection is a container of values defining a single connection between two units
class Connection():

    def __init__(self, exc, affId, effId, weight, delay, n, Q):

        # true if the connection is excitatory, false if it's inhibitory
        if exc:
            self.exc = True
        else:
            self.exc = False

        # afferent and efferent unit ids
        self.aff = affId
        self.eff = effId

        # weight and delay of connection
        self.w = weight
        self.d = delay

        # parameters for sigmoid function
        self.n = n
        self.Q = Q

    def updateAct(self, t):
        val=0.0        
        if t>self.d:
            val=self.w*(Connection.Fa(self.eff.act[int(t-self.d)],self.n,self.Q))
        
        if self.exc== 'True':
            self.aff.sumExc=self.aff.sumExc+val
        else:
            self.aff.sumInh=self.aff.sumInh+val 
   
    def Fa(val, n, Q):
        ret = (val**n) / (Q**n + val**n)
        return ret
 
# basic unit class
class Unit:

    # simulation step for computations
    dT = 0.1
    
    def __init__(self, id):
        self.id = id
        
        self.act = [0,]

        self.sumExc = 0.0
        self.sumInh = 0.0          
         
    def Fa(self, val, n, Q):
        ret = (val**n) / (Q**n + val**n)
        return ret


# fast excitatory unit (oscillatory unit)
class Oscillator(Unit):

    def __init__(self, id):
        Unit.__init__(self, id)

        self.x = random()*0.15
        self.y = random()*0.4 +0.15
        
                
        self.Input = 0.0

    def updateAct(self, stream, t, HPOSC,dist,targbias,distbias):
        # compute the new x and y values based on the formulas:
        # dX/dT = ((-A * x) + (B - x) * (C * f(x) + sumExc + Input + Inoise) - (D * x * f(y)) - (x * sumInh))
        # dY/dT = (E * (x - y))
        # where A=1, B=1, C=20, D=33.3, E=0.15
        # for f: n=4, Q=0.7
                  
        Inoise = random()*0.1
        
        x0 = self.x
        y0 = self.y
        
        # T>CUETIME as cuetime is 0 we implement here t>0
        if self.id in HPOSC and t>0:
            self.sumExc=self.updateBias(x0,self.sumExc,stream,dist,targbias,distbias)

        deltax1 = ((-1 * x0) + (1 - x0) * (20 * self.Fa(x0, 4, 0.7) + self.sumExc + self.Input + Inoise) - (33.3 * x0 * self.Fa(y0, 4, 0.7)) - (x0 * self.sumInh)) * Unit.dT
        deltay1 = (0.15 * (x0 - y0)) * Unit.dT
        
        x1 = x0 + deltax1 * 0.5
        y1 = y0 + deltay1 * 0.5

        deltax2 = ((-1 * x1) + (1 - x1) * (20 * self.Fa(x1, 4, 0.7) + self.sumExc + self.Input + Inoise) - (33.3 * x1 * self.Fa(y1, 4, 0.7)) - (x1 * self.sumInh)) * Unit.dT
        deltay2 = (0.15 * (x1 - y1)) * Unit.dT
        
        x2 = x0 + deltax2 * 0.5
        y2 = y0 + deltay2 * 0.5
        
        deltax3 = ((-1 * x2) + (1 - x2) * (20 * self.Fa(x2, 4, 0.7) + self.sumExc + self.Input + Inoise) - (33.3 * x2 * self.Fa(y2, 4, 0.7)) - (x2 * self.sumInh)) * Unit.dT
        deltay3 = (0.15 * (x2 - y2)) * Unit.dT
        
        x3 = x0 + deltax3
        y3 = y0 + deltay3

        deltax4 = ((-1 * x3) + (1 - x3) * (20 * self.Fa(x3, 4, 0.7) + self.sumExc + self.Input + Inoise) - (33.3 * x3 * self.Fa(y3, 4, 0.7)) - (x3 * self.sumInh)) * Unit.dT
        deltay4 = (0.15 * (x3 - y3)) * Unit.dT
        
        deltax = (deltax1 + 2 * deltax2 + 2 * deltax3 + deltax4)/6
        deltay = (deltay1 + 2 * deltay2 + 2 * deltay3 + deltay4)/6

        self.x += deltax
        self.y += deltay

        self.act.append(self.x)

        # reset synaptic inputs
        self.sumExc = 0.0
        self.sumInh = 0.0

    def Fa(self,val, n, Q):
        ret = (val**n) / (Q**n + val**n)
        return ret
    
    def updateBias(self,x0,sExc,stream,dist,targbias,distbias):
        if dist[stream]==0:
            bias = gauss(targbias,0.1)
            sExc=sExc+ (bias * self.Fa(x0,10,0.1))
            return sExc
        else:
            bias = gauss(distbias,0.1)
            sExc=sExc+ (bias * self.Fa(x0,10,0.1))
            return sExc
    
# slow excitatory unit (leaky integrator)
class Integrator(Unit):

    def __init__(self, id, mod):
        Unit.__init__(self, id)

        self.z = random()*0.05
        self.module = mod
        
    def updateAct(self):
        # compute the new z value based on the formula:
        # dZ/dT = ((-G * z) + sumExc - (z * sumInh))
        # where parameter G and H vary depending on area
        
        if self.module=="LPmap":
            G = 1/30
        elif self.module=="HPmap":
            G = 1/60
        elif self.module=="GW":
            G = 1/250
        elif self.module=="VSWM":
            G = 1/100

        # compute input from paires oscillators, with f(x) parameters n=50 and Q=0.75
        deltaz1 = ((-G * self.z) + self.sumExc - (self.z * self.sumInh)) * Unit.dT
        z1 = self.z + deltaz1 * 0.5

        deltaz2 = ((-G * z1) + self.sumExc - (z1 * self.sumInh)) * Unit.dT
        z2 = self.z + deltaz2 * 0.5

        deltaz3 = ((-G * z2) + self.sumExc - (z2 * self.sumInh)) * Unit.dT
        z3 = self.z + deltaz3

        deltaz4 = ((-G * z3) + self.sumExc - (z3 * self.sumInh)) * Unit.dT

        deltaz = (deltaz1 + 2 * deltaz2 + 2 * deltaz3 + deltaz4)/6
        self.z += deltaz
        
        self.act.append(self.z)

        # reset synaptic inputs
        self.sumExc = 0.0
        self.sumInh = 0.0


# fast inhibitory unit (for lateral and masking inhibitions)
class Interneuron(Unit):

    def __init__(self, id):
        Unit.__init__(self, id)

        self.z = random()*0.15
 
    def updateAct(self):
        # compute the new z value based on the formula:
        # dZ/dT = (-A * z) + (B - z) * (sumExc)
        # where A=1/5, B=1
        
        deltaz1 = ((-0.2 * self.z) + (1 - self.z) * (self.sumExc)) * Unit.dT
        z1 = self.z + deltaz1 * 0.5

        deltaz2 = ((-0.2 * z1) + (1 - z1) * (self.sumExc)) * Unit.dT
        z2 = self.z + deltaz2 * 0.5

        deltaz3 = ((-0.2 * z2) + (1 - z2) * (self.sumExc)) * Unit.dT
        z3 = self.z + deltaz3

        deltaz4 = ((-0.2 * z3) + (1 - z3) * (self.sumExc)) * Unit.dT

        deltaz = (deltaz1 + 2 * deltaz2 + 2 * deltaz3 + deltaz4)/6
        self.z += deltaz
        
        self.act.append(self.z)

        # reset synaptic inputs
        self.sumExc = 0.0
        self.sumInh = 0.0

class MaskingUnit(Unit):

    def __init__(self, id):
        Unit.__init__(self, id)

        self.z = random()*0.05
        self.G = 1/30
        
    def updateAct(self):
        # compute the new z value based on the formula:
        # dZ/dT = ((-G * z) + (B-z)* sumExc
        # where G=1/30 B=1
        
        deltaz1 = ((-(1/30) * self.z) + ((1-self.z)*self.sumExc))* Unit.dT
        z1 = self.z + deltaz1 * 0.5

        deltaz2 = ((-(1/30) * z1) + ((1-z1)*self.sumExc))* Unit.dT
        z2 = self.z + deltaz2 * 0.5

        deltaz3 = ((-(1/30) * z2) + ((1-z2)*self.sumExc))* Unit.dT
        z3 = self.z + deltaz3

        deltaz4 = ((-(1/30) * z3) + ((1-z3)*self.sumExc))* Unit.dT

        deltaz = (deltaz1 + 2 * deltaz2 + 2 * deltaz3 + deltaz4)/6
        self.z += deltaz
        
        self.act.append(self.z)

        # reset synaptic inputs
        self.sumExc = 0.0
        self.sumInh = 0.0     
    
        
# spiking, feeding and gating units in the GW gate
class SpikingUnit(Unit):

    def __init__(self, id):
        Unit.__init__(self, id)

        self.x = random()*0.15
        self.y = (random()*0.05)+0.1
        
    def updateAct(self):
        # compute the new x and y values based on the formulas:
        # dX/dT = ((-A * x) + (B - x) * (C * f(x) + sumExc) - (D * x * f(y)) - (x * sumInh))
        # dY/dT = (E * (x - y))
        # where A=1, B=1, C=20, D=30, E=0.05
        # for f: n=4, Q=0.7

        x0 = self.x
        y0 = self.y

        deltax1 = ((-1 * x0) + (1 - x0) * (20 * self.Fa(x0, 4, 0.7) + self.sumExc) - (30 * x0 * self.Fa(y0, 4, 0.7)) - (x0 * self.sumInh)) * Unit.dT
        deltay1 = (0.05 * (x0 - y0)) * Unit.dT
        
        x1 = x0 + deltax1 * 0.5
        y1 = y0 + deltay1 * 0.5

        deltax2 = ((-1 * x1) + (1 - x1) * (20 * self.Fa(x1, 4, 0.7) + self.sumExc) - (30 * x1 * self.Fa(y1, 4, 0.7)) - (x1 * self.sumInh)) * Unit.dT
        deltay2 = (0.05 * (x1 - y1)) * Unit.dT
        
        x2 = x0 + deltax2 * 0.5
        y2 = y0 + deltay2 * 0.5
        
        deltax3 = ((-1 * x2) + (1 - x2) * (20 * self.Fa(x2, 4, 0.7) + self.sumExc) - (30 * x2 * self.Fa(y2, 4, 0.7)) - (x2 * self.sumInh)) * Unit.dT
        deltay3 = (0.05 * (x2 - y2)) * Unit.dT
        
        x3 = x0 + deltax3
        y3 = y0 + deltay3

        deltax4 = ((-1 * x3) + (1 - x3) * (20 * self.Fa(x3, 4, 0.7) + self.sumExc) - (30 * x3 * self.Fa(y3, 4, 0.7)) - (x3 * self.sumInh)) * Unit.dT
        deltay4 = (0.05 * (x3 - y3)) * Unit.dT
        
        deltax = (deltax1 + 2 * deltax2 + 2 * deltax3 + deltax4)/6
        deltay = (deltay1 + 2 * deltay2 + 2 * deltay3 + deltay4)/6

        self.x += deltax
        self.y += deltay

        self.act.append(self.x)

        # reset synaptic inputs
        self.sumExc = 0.0
        self.sumInh = 0.0

    def Fa(self, val, n, Q):
        ret = (val**n) / (Q**n + val**n)
        return ret

class FeedingUnit(Unit):

    def __init__(self, id):
        Unit.__init__(self, id)

        self.x = random()*0.05
        
    def updateAct(self):
        # compute the new x and y values based on the formula:
        # dX/dT = ((-A * x) + (B - x) * (C * f(x) + sumExc) - (x * sumInh))
        # where A=1, B=1, C=20
        # for f: n=4, Q=0.7

        x0 = self.x
 
        deltax1 = ((-1 * x0) + (1 - x0) * (20 * self.Fa(x0, 4, 0.7) + self.sumExc) - (x0 * self.sumInh)) * Unit.dT
        x1 = x0 + deltax1 * 0.5

        deltax2 = ((-1 * x1) + (1 - x1) * (20 * self.Fa(x1, 4, 0.7) + self.sumExc) - (x1 * self.sumInh)) * Unit.dT
        x2 = x0 + deltax2 * 0.5
        
        deltax3 = ((-1 * x2) + (1 - x2) * (20 * self.Fa(x2, 4, 0.7) + self.sumExc) - (x2 * self.sumInh)) * Unit.dT
        x3 = x0 + deltax3

        deltax4 = ((-1 * x3) + (1 - x3) * (20 * self.Fa(x3, 4, 0.7) + self.sumExc) - (x3 * self.sumInh)) * Unit.dT
        
        deltax = (deltax1 + 2 * deltax2 + 2 * deltax3 + deltax4)/6

        self.x += deltax

        self.act.append(self.x)

        # reset synaptic inputs
        self.sumExc = 0.0
        self.sumInh = 0.0

    def Fa(self, val, n, Q):
        ret = (val**n) / (Q**n + val**n)
        return ret
        
class GateUnit(Unit):

    def __init__(self, id):
        Unit.__init__(self, id)

        self.z = random()*0.05
 
    def updateAct(self):
        # compute the new z value based on the formula:
        # dZ/dT = ((-G * z) + sumExc - (z * sumInh))
        # where G=1/50
        
        deltaz1 = ((-(1/50) * self.z) + self.sumExc - (self.z * self.sumInh)) * Unit.dT
        z1 = self.z + deltaz1 * 0.5

        deltaz2 = ((-(1/50) * z1) + self.sumExc - (z1 * self.sumInh)) * Unit.dT
        z2 = self.z + deltaz2 * 0.5

        deltaz3 = ((-(1/50) * z2) + self.sumExc - (z2 * self.sumInh)) * Unit.dT
        z3 = self.z + deltaz3

        deltaz4 = ((-(1/50) * z3) + self.sumExc - (z3 * self.sumInh)) * Unit.dT

        deltaz = (deltaz1 + 2 * deltaz2 + 2 * deltaz3 + deltaz4)/6
        self.z += deltaz
        
        self.act.append(self.z)

        # reset synaptic inputs
        self.sumExc = 0.0
        self.sumInh = 0.0

    def Fa(self, val, n, Q):
        ret = (val**n) / (Q**n + val**n)
        return ret
        
class BlockingUnit(Unit):

    def __init__(self, id):
        Unit.__init__(self, id)

        self.z = random()*0.05
 
    def updateAct(self):
        # compute the new z value based on the formula:
        # dZ/dT = (-A * z) + (B - z) * (C * f(z) + sumExc)
        # where A=1, B=1, C=20
        # for f: n=4, Q=0.7
        x0=self.z
        
        deltaz1 = ((-1 * x0) + (1 - x0) * (20 * self.Fa(x0, 4, 0.7) + self.sumExc)) * Unit.dT
        z1 = x0 + deltaz1 * 0.5

        deltaz2 = ((-1 * z1) + (1 - z1) * (20 * self.Fa(z1, 4, 0.7) + self.sumExc)) * Unit.dT
        z2 = self.z + deltaz2 * 0.5

        deltaz3 = ((-1 * z2) + (1 - z2) * (20 * self.Fa(z2, 4, 0.7) + self.sumExc)) * Unit.dT
        z3 = self.z + deltaz3

        deltaz4 = ((-1 * z3) + (1 - z3) * (20 * self.Fa(z3, 4, 0.7) + self.sumExc)) * Unit.dT

        deltaz = (deltaz1 + 2 * deltaz2 + 2 * deltaz3 + deltaz4)/6
        self.z += deltaz
        
        self.act.append(self.z)

        # reset synaptic inputs
        self.sumExc = 0.0
        self.sumInh = 0.0

    def Fa(self, val, n, Q):
        ret = (val**n) / (Q**n + val**n)
        return ret

class Inputunit(Unit):
    def __init__(self, id):
        Unit.__init__(self, id)
        
        self.x = random()*0.15
        self.y = random()*0.4 +0.15
        
                
        self.Input = 0.0

    def updateAct(self,stream,t,tstream,tstim):
        # compute the new x and y values based on the formulas:
        # dX/dT = ((-A * x) + (B - x) * (C * f(x) + Input ) - (D * x * f(y)) - (x * sumInh))
        # dY/dT = (E * (x - y))
        # where A=1, B=1, C=20, D=33.3, E=0.20
        # for f: n=4, Q=0.7
   
#       Inoise = random()*0.1
       tm=t
       nstream=stream 
       tarstream=tstream
       tarstim=tstim
       
       Inpt=Inputunit.updateInput(nstream,tm,tarstream,tarstim)
       
       
       x0 = self.x
       y0 = self.y

       deltax1 = ((-1 * x0) + ((1 - x0) * (20 * self.Fa(x0, 4, 0.7) + Inpt)) - (33.3 * x0 * self.Fa(y0, 4, 0.7)) - (x0 * self.sumInh)) * Unit.dT
       deltay1 = (0.20 * (x0 - y0)) * Unit.dT
        
       x1 = x0 + deltax1 * 0.5
       y1 = y0 + deltay1 * 0.5

       deltax2 = ((-1 * x1) + ((1 - x1) * (20 * self.Fa(x1, 4, 0.7) + Inpt)) - (33.3 * x1 * self.Fa(y1, 4, 0.7)) - (x1 * self.sumInh)) * Unit.dT
       deltay2 = (0.20 * (x1 - y1)) * Unit.dT
        
       x2 = x0 + deltax2 * 0.5
       y2 = y0 + deltay2 * 0.5
        
       deltax3 = ((-1 * x2) + ((1 - x2) * (20 * self.Fa(x2, 4, 0.7) + Inpt)) - (33.3 * x2 * self.Fa(y2, 4, 0.7)) - (x2 * self.sumInh)) * Unit.dT
       deltay3 = (0.20 * (x2 - y2)) * Unit.dT
        
       x3 = x0 + deltax3
       y3 = y0 + deltay3

       deltax4 = ((-1 * x3) + ((1 - x3) * (20 * self.Fa(x3, 4, 0.7) + Inpt)) - (33.3 * x3 * self.Fa(y3, 4, 0.7)) - (x3 * self.sumInh)) * Unit.dT
       deltay4 = (0.20 * (x3 - y3)) * Unit.dT
        
       deltax = (deltax1 + 2 * deltax2 + 2 * deltax3 + deltax4)/6
       deltay = (deltay1 + 2 * deltay2 + 2 * deltay3 + deltay4)/6

       self.x += deltax
       self.y += deltay

       self.act.append(self.x)

        # reset synaptic inputs
       self.sumExc = 0.0
       self.sumInh = 0.0
     
    def Fa(self, val, n, Q):
        ret = (val**n) / (Q**n + val**n)
        return ret
    
    def updateInput(stream,te,tarstream,tarstim):
        if te-tarstream<= tarstim and te>= tarstream:
                ret=0.3
                return ret
        else:
            ret=0.0
            return ret
