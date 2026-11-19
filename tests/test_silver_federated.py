from pathlib import Path
import unittest
from medgemma_fl.experiments.run_silver_federated import command


class SilverLauncherTests(unittest.TestCase):
    def test_disjoint_experiment_plan_preserves_local_effective_batch(self):
        for method, group, expected_accum in [('fedavg','0,1',2),('fedlc','3',4)]:
            cmd=command(Path('data'),Path('run'),method,group,60,1,20261117,False,None)
            for flag,value in [('--fl_method',method),('--seed','20261117'),
                               ('--num_rounds','60'),('--num_train_epochs','1'),
                               ('--gradient_accumulation_steps',str(expected_accum)),
                               ('--effective_batch_size','4')]:
                self.assertEqual(cmd[cmd.index(flag)+1],value)
            self.assertIn('--weighted_labels',cmd)
            self.assertIn('--no-local_eval',cmd)
            self.assertNotIn('--max_steps',cmd)
            self.assertEqual(cmd[cmd.index('--gpu')+1],','.join([f'[{group}]']*3))
            self.assertEqual(cmd[cmd.index('--save_global_rounds')+1:cmd.index('--weighted_labels')],
                             ['2','5','10','20','30','50','60'])

    def test_smoke_is_not_the_formal_budget(self):
        cmd=command(Path('data'),Path('run'),'fedlc','3',60,1,20261117,True,None)
        self.assertEqual(cmd[cmd.index('--num_rounds')+1],'1')
        self.assertEqual(cmd[cmd.index('--max_steps')+1],'1')


if __name__=='__main__':unittest.main()
