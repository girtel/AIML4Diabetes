import numpy as np
from simglucose.analysis.risk import risk_index, magni_RI

bg_opt=112.517 #The risk function has a minimun here, ris(112.517)=0

def stepReward3(BG,CGM, CHO, insulin, done=False):
   bg_current=BG
   LBGI, HBGI, RI = risk_index([bg_current], 1)
   mRI = magni_RI([bg_current], 1)
   if bg_current >= 70 and bg_current <= 180:
      reward=1
   else: reward=0
   if done:
      reward=-100
   return reward

def stepReward3_eval(BG,CGM, CHO, insulin, done=False):
   bg_current=BG
   LBGI, HBGI, RI = risk_index([bg_current], 1)
   mRI = magni_RI([bg_current], 1)
   if bg_current >= 70 and bg_current <= 180:
      reward=1
   else: reward=0
   if done:
      reward=-100
   print("Action:", insulin,";CHO:",CHO,";reward:", reward , ";BG:", BG, ";CGM:", CGM, ";RI:" , RI, ";LBGI:", LBGI, ";HBGI:", HBGI, ";mRI:", mRI)
   return reward
