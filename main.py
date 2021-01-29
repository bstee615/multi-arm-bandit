import numpy as np

k=5
n=100
p=np.random.rand(k)
bandits = np.array([np.random.binomial(1, p[i], n) for i in range(k)])
print(f'k-armed bandit: {k=}, {n=}, {p=}')

def explore_then_exploit(n_explore = 50, n_exploit = 50):
  """
  Explore only for some times, then calculate q_a and use that to exploit (an approximation of) the best bandit.
  """
  total_n = n_explore + n_exploit

  # Do exploration
  def explore():
    """
    Sample a random bandit to pull.
    """
    a = np.random.choice(range(bandits.shape[0]))
    return a, np.random.choice(bandits[a])
  exploration = np.array([explore() for i in range(n_explore)])
  exploration_reward = exploration.sum(axis=0)[1]

  sums = np.zeros((bandits.shape[0],))
  lens = np.zeros((bandits.shape[0],))
  for a, r in exploration:
    sums[a] += r
    lens[a] += 1
  q_a = np.divide(sums, lens, out=np.zeros_like(sums), where=lens != 0)

  # Exploit the best bandit
  def exploit(q_a):
    """
    Choose argmax from q_a.
    If there is a tie, choose randomly.
    """
    maxes = np.argwhere(q_a == np.amax(q_a)).T[0]
    a = np.random.choice(maxes)
    return a, np.random.choice(bandits[a])
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

explore_then_exploit(1, 100)
