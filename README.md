### Final Demo

`Quadruped Trained with PPO for 10k episodes in action playing the game of soccer` 
[<img src="https://img.youtube.com/vi/uRmXBiZup3M/maxresdefault.jpg" width="50%">](https://youtu.be/uRmXBiZup3M)

### Instructions to use this repo :

#### Prerequisites
* conda (This project was implemented under the `base` environment but can also be executed under a custom conda environment)
* gym (v0.15.3) ( pip install gym==0.15.3) 
* mujoco (version 200) ( https://medium.com/@chinmayburgul/setting-up-mujoco200-on-linux-16-04-18-04-38e5a3524c85 )


1. Install deepmind control suite 

```bash
theretrovenger@asus_rog$~: cd dm_control/
theretrovenger@asus_rog$~: pip3 install -e .
```

**NOTE:** Replace the default ```quadruped.py``` file & ```quadruped.xml``` from the ```site-packages``` installation of ```dm_control``` file with the provided ```quadruped.py``` file && ```quadruped_soccer.xml``` file from this folder    

2. Install `dm2gym` python module (it is a deepmind control suite -> gym wrapper (v0.2.0 (latest at the time of writing) )

```bash
theretrovenger@asus_rog$~: pip3 install dm2gym==0.2.0 
```

**NOTE:** Replace the ```dm2gym``` folder in the ```anaconda3/lib/python3.7/site-packages``` with the provided ```dm2gym``` folder (this contains a package fix) 

3. Next, Install the baselines package which will be used with ```a2c-ppo-acktr``` package
If there is any error during installation, install tensorflow, tensorflow-gpu

```bash
theretrovenger@asus_rog$~: cd .
theretrovenger@asus_rog$~: git clone https://github.com/openai/baselines.git
theretrovenger@asus_rog$~: cd baselines/
theretrovenger@asus_rog$~: pip3 install -e .
```
4. Run RL Algorithms from the ```pytorch-a2c-ppo-acktr-gail-master```  folder. 
```bash
theretrovenger@asus_rog$~: cd ./pytorch-a2c-ppo-acktr-gail-master
```
* Training :

    For A2C -
    
    `python main.py --env-name dm.quadruped.soccer --num-processes 8 --num-steps 128 --num-mini-batch 4`


    For PPO -
    
    `python main.py --env-name "dm.quadruped.soccer" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01`
    
* Testing & Visualizing :


    `python nenjoy.py --load-dir trained_models/<algo> --env-name "dm.quadruped.soccer"`
    
NOTE: Please use the below author credits for further use since most of the project code reference was made from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail 

```Citation :
@misc{pytorchrl,
  author = {Kostrikov, Ilya},
  title = {PyTorch Implementations of Reinforcement Learning Algorithms},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{ https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail  }},
}```
