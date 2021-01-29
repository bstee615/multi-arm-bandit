import numpy as np

k=5
n=100
p=np.random.rand(k)
bandits = np.array([np.random.binomial(1, p[i], n) for i in range(k)])
print(f'k-armed bandit: {k=}, {n=}, {p=}')

def explore():
  def sample():
    return np.random.choice(range(bandits.shape[0]))
  a = sample()
  return a, np.random.choice(bandits[a])

def exploit(q_a):
  a = np.argmax(q_a)
  return a, np.random.choice(bandits[a])

def explore_then_exploit(n_explore = 100, n_exploit = 100):
  """
  Explore 10 times, then calculate Q_a and use that to exploit
  """
  total_n = n_explore + n_exploit

  # Do exploration
  exploration = np.array([explore() for i in range(n_explore)])
  exploration_reward = exploration.sum(axis=0)[1]

  sums = np.zeros((bandits.shape[0],))
  lens = np.zeros((bandits.shape[0],))
  for a, r in exploration:
    # print(a, r)
    sums[a] += r
    lens[a] += 1
  q_a = sums / lens

  # Exploit the best bandit
  exploitation = np.array([exploit(q_a) for i in range(n_exploit)])
  exploitation_reward = exploitation.sum(axis=0)[1]

  # Report results
  sum_reward = exploration_reward + exploitation_reward
  print(f'For {n_explore=} {n_exploit=} {total_n=}:')
  print(f'{q_a=}')
  print(f'{exploration_reward=} {exploitation_reward=} {sum_reward=}')

  all_rewards = np.concatenate((exploration[:,1], exploitation[:,1]))
  assert all_rewards.shape[0] == total_n
  return all_rewards

explore_then_exploit()
