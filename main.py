from net import NeuralNet
from PPO import PPO
import gymnasium as gym
import argparse

def args_for_run():
    pares = argparse.ArgumentParser(description="traning or testing model")
    pares.add_argument('mode', help='', default='train')
    pares.add_argument('--path_to_actor', dest='actor_model', type=str, default='')
    pares.add_argument('--path_to_critic', dest='critic_model', type=str, default='')
    args = pares.parse_args()
    return args

if __name__ == "__main__":
    args= args_for_run()
    env = gym.make("CartPole-v1",render_mode = 'human' if args.mode == 'test' else None)
    ppo = PPO(NeuralNet, env)
    load =  args.actor_model != "" and args.critic_model != ""
    if args.mode == 'train':
        if load:
            ppo.load(args.actor_model,args.critic_model)
            print("load successful")
        ppo.learn(2_000_000)
    elif args.mode == 'test':
        if args.actor_model:
            ppo.load(args.actor_model)
            for i in ppo.show_results():
                print(i)
        else:
            raise FileNotFoundError(
                'you must specif actor path'
            )