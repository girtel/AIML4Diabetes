# AIML4Diabetes

##For PPO-RNN

Install environment

Open trainagentpar.py and change parameters and hyperparameters.

You can try only change some parameters in line 92 and 155
```
patient_name='adult#010', #Patient name: adult#001-010, adolescent#001-010 and child#001-010
reward_fun=stepReward3, #Change to your reward function
normalize=True, #Normalize input
sequence=15, #Insulin injection interval, 1 seq = 3 mins
harrison_benedict=True)), #Use Harrison-Benedict's meal schedule
370 #Step time limit
```
