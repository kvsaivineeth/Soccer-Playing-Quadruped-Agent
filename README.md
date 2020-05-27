### Demo

[![Watch the video]()](https://youtu.be/uRmXBiZup3M)

### To run this project

Note: 
      Have \ conda / preinstalled
      Have gym >=0.15.3

1. Install deepmind control suite 

```
cd dm_control/
pip3 install -e .
```
Replace the default ```quadruped.py``` file & ```quadruped.xml``` from the ```site-packages``` installation of ```dm_control``` file with the provided ```quadruped.py``` file && ```quadruped_soccer.xml``` file from this folder    

2. Install dm2gym (deepmind control suite -> gym wrapper, Make sure it is version 0.2.0 (latest) )

```
pip3 install dm2gym 
```

Replace the ```dm2gym``` folder in the ```anaconda3/lib/python3.7/site-packages``` with the provided ```dm2gym``` folder (this contains a package fix) 

3. Next, Install the baselines package which will be used with ```a2c-ppo-acktr``` package
If there is any error during installation, install tensorflow, tensorflow-gpu
```
cd .
git clone https://github.com/openai/baselines.git
cd baselines/
pip3 install -e .
```
4. Run Algos from the ```pytorch-a2c-ppo-acktr-gail-master```  folder. It can be done 

(i) Training :

    For A2C
    
    python main.py --env-name dm.quadruped.soccer --num-processes 8 --num-steps 128 --num-mini-batch 4


    For PPO
    
    python main.py --env-name "dm.quadruped.soccer" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01
    
(ii) Testing & Visualizing :


    python nenjoy.py --load-dir trained_models/<algo> --env-name "dm.quadruped.soccer"
    


For Citation :
@misc{pytorchrl,
  author = {Kostrikov, Ilya},
  title = {PyTorch Implementations of Reinforcement Learning Algorithms},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{ https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail  }},
}
