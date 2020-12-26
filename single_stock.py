import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from stable_baselines3.a2c.a2c import A2C
matplotlib.use('Agg')
import datetime

from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.environment import EnvSetup
from finrl.model.models import DRLAgent
from single_stock_env import SingleStockEnv
from finrl.trade.backtest import BackTestStats, BaselineStats, BackTestPlot

import os

if __name__ == "__main__":     
    if not os.path.exists("./" + config.DATA_SAVE_DIR):
        os.makedirs("./" + config.DATA_SAVE_DIR)
    if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
        os.makedirs("./" + config.TRAINED_MODEL_DIR)
    if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
        os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
    if not os.path.exists("./" + config.RESULTS_DIR):
        os.makedirs("./" + config.RESULTS_DIR)

    '''
    df = YahooDownloader(start_date = '2009-01-01',
        end_date = '2020-12-01',
        ticker_list = ['AAPL']).fetch_data()
    
    df.sort_values(['date','tic'],ignore_index=True)

    df = FeatureEngineer(df.copy(),
                    use_technical_indicator=True,
                    tech_indicator_list = config.TECHNICAL_INDICATORS_LIST,
                    use_turbulence=False,
                    user_defined_feature = True).preprocess_data()
    
    df.to_csv('data/AAPL')
    '''
    
    df = pd.read_csv('data/akbank.csv')

    train = data_split(df, '2009-01-01','2019-01-01')
    trade = data_split(df, '2019-01-01','2020-12-01')


    print(train.head())
    print(trade.head())

    stock_dimension = len(train.tic.unique())
    #state_space = 1 + 2*stock_dimension + len(config.TECHNICAL_INDICATORS_LIST)*stock_dimension
    state_space = 1 + 2*stock_dimension + len(config.TECHNICAL_INDICATORS_LIST)*stock_dimension + 4*stock_dimension

    env_setup = EnvSetup(stock_dim = stock_dimension,
        state_space = state_space,
        hmax = 500,
        initial_amount = 10000,
        transaction_cost_pct = 0.002)

    env_train = env_setup.create_env_training(data = train, env_class = SingleStockEnv)
    env_trade, obs_trade = env_setup.create_env_trading(data = trade, env_class = SingleStockEnv) 

    '''
    print("==============A2C Model Training===========")
    agent = DRLAgent(env = env_train)
    now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
    a2c_params_tuning = {'n_steps':5, 
                'ent_coef':0.005, 
                'learning_rate':0.0002,
                'verbose':0,
                'timesteps':20000}
    
    model_a2c = agent.train_A2C(model_name = "A2C_{}".format(now), model_params = a2c_params_tuning)
    #model_a2c = A2C.load('trained_models/A2C_20201220-16h41.zip')

    print("==============DDPG Model Training===========")
    agent = DRLAgent(env = env_train)
    now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
    ddpg_params_tuning = {
                        'batch_size': 128,
                            'buffer_size':100000, 
                        'learning_rate':0.0003,
                            'verbose':0,
                            'timesteps':30000}
    model_ddpg = agent.train_DDPG(model_name = "DDPG_{}".format(now), model_params = ddpg_params_tuning)
    '''

    print("==============PPO Model Training===========")
    agent = DRLAgent(env = env_train)
    now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
    ppo_params_tuning = {'n_steps':128, 
                        'batch_size': 64,
                        'ent_coef':0.005, 
                        'learning_rate':0.00025,
                        'verbose':0,
                        'timesteps':50000}
    model_ppo = agent.train_PPO( model_name = "PPO_{}".format(now), model_params = ppo_params_tuning)

    print("==============Start Trading===========")
    env_trade, obs_trade = env_setup.create_env_trading(data = trade,
        env_class = SingleStockEnv,
        turbulence_threshold=250) 

    df_account_value, df_actions = DRLAgent.DRL_prediction(model=model_ppo,
        test_data = trade,
        test_env = env_trade,
        test_obs = obs_trade)
    

    df_account_value.to_csv("./"+config.RESULTS_DIR+"/df_account_value_"+now+'.csv')
    df_actions.to_csv("./"+config.RESULTS_DIR+"/df_actions_"+now+'.csv')

    print("==============Get Backtest Results===========")
    perf_stats_all = BackTestStats(df_account_value)
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_csv("./"+config.RESULTS_DIR+"/perf_stats_all_"+now+'.csv')
