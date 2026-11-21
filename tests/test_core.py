import unittest
import torch
from pathlib import Path
from argparse import Namespace
from ocrl.observations import ObservationAdapter, load_config, portable_arguments
from ocrl.storage.replay_buffer import ReplayBuffer
from ocrl.models.factory import CreateWorlModelInstance

class CoreTests(unittest.TestCase):
    def test_metadata_paths_are_portable(self):
        args=Namespace(config=Path('configs/default.json').resolve(), seed=42)
        saved=portable_arguments(args)
        self.assertEqual(saved['config'], 'configs/default.json')
        self.assertFalse(Path(saved['config']).is_absolute())
        self.assertEqual(saved['seed'],42)

    def test_wrapped_replay_is_chronological(self):
        b=ReplayBuffer([1,1,1],5,'cpu')
        for i in range(8):b.insert([torch.tensor([[float(i)]]),torch.zeros(1,1),torch.zeros(1,1)])
        states=next(b.mini_batch_generator(3,1,100))[0]
        self.assertTrue(torch.all(states[:,1:]-states[:,:-1]==1))
        self.assertGreaterEqual(states.min().item(),3)
    def test_sequences_do_not_cross_reset(self):
        b=ReplayBuffer([1,1,1],8,'cpu')
        for i in range(8):b.insert([torch.tensor([[float(i)]]),torch.zeros(1,1),torch.tensor([[i==3]])])
        states=next(b.mini_batch_generator(3,1,100))[0].squeeze(-1)
        self.assertFalse(((states[:,0]<=3)&(states[:,-1]>3)).any())
    def test_insufficient_history_is_explicit(self):
        b=ReplayBuffer([1,1,1],8,'cpu');b.insert([torch.zeros(1,1)]*3)
        with self.assertRaises(ValueError):next(b.mini_batch_generator(3,1,1))
    def test_observation_layout(self):
        cfg=load_config();a=ObservationAdapter(cfg,'cpu')
        state=torch.tensor(cfg['mean_state_action'][:96]).repeat(2,1)
        actions=torch.ones(2,29);commands=torch.tensor([[1.,2.,3.]]).repeat(2,1)
        obs=a.policy_obs(state,actions,commands)['policy']
        self.assertEqual(tuple(obs.shape),(2,99))
        torch.testing.assert_close(obs[:,:67],torch.zeros(2,67))
        torch.testing.assert_close(obs[:,67:96],actions)
        torch.testing.assert_close(obs[:,-3:],commands)
    def test_model_forward_backward(self):
        m=CreateWorlModelInstance(load_config())
        x=torch.zeros(2,1,125);h=torch.zeros(2,2,256)
        pred,_=m(x,h,predict=True,x_prev=x[:,:,:96])
        loss=m.mse_loss(pred[:,0],torch.ones(2,96));loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))

if __name__=='__main__':unittest.main()
