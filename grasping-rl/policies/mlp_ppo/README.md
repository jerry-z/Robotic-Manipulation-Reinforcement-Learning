To train the PPO model: 

`python train_roam_hand.py`

The Tensorboard log files, the running mean and std for normalization and the trained model are saved in the /logs directory with time as the folder name

To plot the training curve via Tensorboard, cd to the folder mentioned above and do

`tensorboard --logdir=.`

and go to

`http://localhost:6006` in the browser.


To load and run, use the folder name as the argument. For example:

`python load_run_model.py 07-02-2019-11-35-27_1`
