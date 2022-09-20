# DRL_FinalProject_Group24 
Lisa Golla & Hannah Köster 

A project for the course Deep Reinforcement learning. SS 2022   
Prof. Dr. Elia Bruni & Prof. Dr. Gordon Pipa  
Universität Osnabrück   


## PAIRED algorithm - Working on task of Automated Map Generation with DRL

Overall, our task can be summarized as the following. We found a relevant paper which has implemented an unsupervised gridworld generation by introducing the Protagonist Antagonist Induced Regret Environment Design (PAIRED) algorithm for automatic environment generation. We used the implementation of the paper and adapted it to our needs, e.g. there is no goal state in the environment that the agent has to reach, rather his task is to find an effective and robust solution for exploring the unknown and creating a map of the environment in the process. Therefore, we had to adapt the reward, goal statement and visualization part as well as the metrics that are logged in the code section. How we adapted goal and reward in our experiment will be explained in  more detail in the paper. We applied two optimized variants of the PAIRED algorithm, flexible PAIRED and budget PAIRED, as well as the combination so-called flexible b-PAIRED. Moreover, we also tested how the vanilla PAIRED algorithm performs with and without continuously negative reward.

This paper makes the following contributions:
- Testing how the PAIRED algorithm performs regarding the task of robotic mapping
- Comparing the performance of different optimized versions of PAIRED
- Testing the influence of continuous negative reward on the performance 
- Gives rise to a learned policy which can be transferred to more complex and large environments
- Introduces useful metrics to evaluate the performance 



For more details about our work check out [our paper](Applying_PAIRED_to_Automated_Map_Generation.pdf). 
[![A video summing up the crucial points can also be found.](Video_RL_Projekt_NEW.mp4)](Video_RL_Projekt_NEW.mp4)


## How to use this Github Repo 
  ### How to execute the code 
  What is important to note here is that the parameters correspond to the following definition:   
    - `flexible_protagonist`: applies a PAIRED optimized approach called flexible PAIRED (True / False)  
    - `block_budget_weight`: applies a PAIRED optimized approach called budget PAIRED (determines the weight we give the budget)  
    
  `python train_adversarial__env.py --debug --root_dir=*DirectoryYouWantToStoreTheResultsIn*  --random_seed=42 --flexible_protagonist=True  --block_budget_weight=0 --num_train_steps=1000`
    
  ### How to display data in Tensorboard
  - go into the specific folder that contains the results you are interested in. The folders are named after the algorithm version used except for `no_neg_reward` which contains the logged data for PAIRED with reward version R2 (which does not have the negative reward each time step).
    
    → if you want to access the tensorboard data for flexible PAIRED, your path would be for example `DRL_FinalProject_Group24\social_rl\tmp\flexible`
  
  - start the tensorboard using `tensorboard --logdir train`
  
  ### Where to find videos of the agent's behavior. 
  Depending on which trial you interested in, just exchange "paired" by another trial.  
  `DRL_FinalProject_Group24\social_rl\tmp\paired\vids`
 


## Organization
Along our journey of finishing this lovely project, we had certain meetings with the tutor of the course, to clarify questions and to represent out work progress. A pdf file containing a meeting summary is also attached to the repo.


## Future Iterations 
We allow the usage of our work to be used for future iterations.
