"""Plot saved evaluation or training data without rerunning simulation."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('run',type=Path)
a=p.parse_args()
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
if (a.run/'predictions.npz').exists():
    data=np.load(a.run/'predictions.npz')
    actual,pred=data['actual'],data['predicted']
    t=np.arange(1,len(pred)+1)*float(data['dt'])
    groups=[('Linear velocity',slice(0,3)),('Angular velocity',slice(3,6)),('Projected gravity',slice(6,9)),('Joint position',slice(9,38)),('Joint velocity',slice(38,67)),('Joint torque',slice(67,96))]
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for ax,(name,sl) in zip(axes.flat,groups):
        ax.plot(t,np.sqrt(np.mean((pred[:,sl]-actual[:,sl])**2,axis=1)),color='#176b87',lw=2)
        ax.set(title=name,xlabel='Forecast horizon (s)',ylabel='Normalized RMSE');ax.grid(alpha=.2)
    fig.suptitle('World-model forecast error · one uninterrupted trajectory')
    fig.savefig(a.run/'prediction_error.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(10,8),layout='constrained')
    for ax,index,name in zip(axes,[0,6,9],['World-frame forward velocity','Projected gravity x','Joint position 0']):
        ax.plot(t,actual[:,index],label='Simulation',color='#176b87');ax.plot(t,pred[:,index],'--',label='World model',color='#dd6b20')
        ax.set(title=name,xlabel='Forecast horizon (s)',ylabel='Normalized state');ax.legend();ax.grid(alpha=.2)
    fig.savefig(a.run/'predictions.png',dpi=160);plt.close(fig)
if (a.run/'metrics.csv').exists():
    data=np.atleast_1d(np.genfromtxt(a.run/'metrics.csv',delimiter=',',names=True))
    if 'vx_rmse' in data.dtype.names:
        fig,axes=plt.subplots(2,1,figsize=(10,6),layout='constrained')
        axes[0].plot(data['step'],data['simulator_reward'],color='#176b87');axes[0].set(ylabel='Simulator reward')
        for key in ['vx_rmse','vy_rmse','yaw_rate_rmse']:axes[1].plot(data['step'],data[key],label=key)
        axes[1].legend();axes[1].set(xlabel='Simulation step',ylabel='Tracking error (m/s or rad/s)')
        fig.savefig(a.run/'policy_metrics.png',dpi=160);plt.close(fig)
    else:
        fig,axes=plt.subplots(2,1,figsize=(10,6),layout='constrained')
        axes[0].plot(data['iteration'],data['world_model_loss']);axes[0].set(ylabel='World-model training loss')
        for key in ['real_reward','imagined_reward']:axes[1].plot(data['iteration'],data[key],label=key)
        axes[1].legend();axes[1].set(xlabel='Iteration',ylabel='Custom reward')
        fig.savefig(a.run/'training.png',dpi=160);plt.close(fig)
