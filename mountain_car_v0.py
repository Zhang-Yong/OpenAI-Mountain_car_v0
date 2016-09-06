# This code works using the cross-entropy method.
#Modify  from cartpole version of https://github.com/cgnicholls/reinforcement-learning/blob/master/cartpole/crossentropy.py
# Zhang Yong, Sep-05-2016
# Mail: 1930748422@qq.com
import numpy as np
import random
import gym
import math
env = gym.make('MountainCar-v0')

# Play with cross entropy agent, with given parameter vector
def crossEntropyAgent(num_episodes, max_episode_length, theta):
    rewards = []
    for i_episode in range(num_episodes):
        episode_reward = 0
        observation = env.reset()
        for t in range(max_episode_length):
            env.render()
            action = sampleAction(observation, theta)
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                rewards.append(episode_reward)
                print("Reward for episode:", episode_reward)
                break

def trainCrossEntropyAgent(num_episodes, batch_size, max_episode_length, elite_fraction):
    # Initialise mu and sigma.
    mu = [0,0]
    sigma = [20,20]         # Yong, large sigma learns much faster
    env.monitor.start('/tmp/MountainCar-v0-experiment-1',force=True)
    for i_episode in range(num_episodes):
        print("Episode: ", i_episode)
        batch_rewards = []
        batch_theta = []
        for i_batch in range(batch_size):
            theta = sampleTheta(mu, sigma)
            observation = env.reset()
            batch_reward = 0
            for t in xrange(max_episode_length):
                #env.render()
                action = sampleAction(observation, theta)
                observation, reward, done, info = env.step(action)
                batch_reward += reward
                if done:
                    batch_rewards.append(batch_reward)
                    batch_theta.append(theta)
                    #print ("done, batch_rewards", batch_rewards)
                    break
        
        # Print the average reward
        print("Average rewards in training", np.mean(batch_rewards))
        #print ("batch_theta", batch_theta)
        # Now keep the top elite_fraction fraction of parameters theta, as
        # measured by reward
        
       # if (batch_reward != -200):
        indices = np.argsort(np.array(batch_rewards))
        indices = indices[::-1]
         
        elite_set = []
        cull_num = int(elite_fraction * len(indices))
        # select Top N samples here instead of percentage
        #cull_num = elite_fraction
        for i in xrange(cull_num):
            elite_set.append(batch_theta[indices[i]])
        #print "elite set", elite_set
        # Now fit a diagonal Gaussian to this sample set, and repeat.
        [mu, sigma2] = fitGaussianToSamples(elite_set)
        sigma = np.sqrt(sigma2)
        #print ("mu,sigma", mu,sigma)
        #else:
#             mu = [0,0]
#             sigma = [20,20]   
            
    env.monitor.close()
    # Finally, return the mean we find
    return mu

def sampleTheta(mu, sigma):
    #print ("sigma, mu", sigma, mu)
    return np.random.randn(1,2) * sigma + mu

# Actions are 3 discret, generate value of 0, 1, 2
def sampleAction(observation, theta):
    result = np.dot(observation, np.transpose(theta))
    result = int(sigmoid(result)*4)
    return result
   
        
# Given a matrix whose rows are samples from a multivariate gaussian
# distribution with diagonal covariance matrix, we compute the maximum
# likelihood mean and covariance matrix. In fact we just return the diagonal of
# the covariance matrix.
def fitGaussianToSamples(samples):
    M = np.matrix(np.array(samples))
    Mshape = np.shape(M)
    numSamples = Mshape[0]
    numVariables = Mshape[1]

    # For each variable, we compute the mean and variance of the samples.
    mu = []
    sigma2 = []

    for i in xrange(numVariables):
        variableI = M[:,i]
        mu.append(np.mean(variableI))
        sigma2.append(np.var(variableI))
    return [mu, sigma2]

#generate [0,1]
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

    
    
#num_episodes, batch_size, max_episode_length, elite_fraction
time_step_limit = 200
#bigger episodes and batch size get more awards
theta = trainCrossEntropyAgent(30, 1000, time_step_limit, 0.1)

print("Theta after training", theta)

crossEntropyAgent(50, time_step_limit, theta)