# AIML4Diabetes

## For PPO-RNN

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
Command for training:
```
python trainagentC1.py --root_dir log/patient --alsologtostderr > log.txt 2> logerror.txt 
```
For evaluation in evalagent.py also same as above. You can use trained policies in ppo-rnn-policies folder.
Command for evaluation:
```
python evalagent.py --root_dir logeval/patient --saved_dir ppo-rnn-policies/XXX/policy_saved_model/policy_000006400/ > log.txt 2> logerror.txt 
```

## For BB and PID

You can set how many random seeds and days to run in BBGreedy.py and PIDGreedy.py


Created by Phuwadol Viroonluecha.
Developed and contributed by Phuwadol Viroonluecha, Eateban Egea Lopez and Jose Santa.
