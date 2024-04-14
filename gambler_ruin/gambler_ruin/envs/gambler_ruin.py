import numpy as np
from random import randint
import gymnasium as gym
from gymnasium import spaces

class GamblerRuin(gym.Env):
    def __init__(self, nextStateTable, rewardsTable):
        super(GamblerRuin, self).__init__()
        self.__version__ = "0.1.0"
        # print("CMM Finite MDP - Version {}".format(self.__version__))
        self.nextStateTable = nextStateTable
        self.rewardsTable = rewardsTable  # expected rewards

        self.current_observation_or_state = 0

        # (S, A, nS) = self.nextStateProbability.shape #should not require nextStateProbability, which is often unknown
        self.S = rewardsTable.shape[0]  # number of states
        self.A = rewardsTable.shape[1]  # number of actions

        self.possible_states = np.arange(self.S)

        # initialize possible states and actions
        # we need to indicate only valid actions for each state
        # create a list of lists, that indicates for each state, the list of allowed actions
        self.possible_actions_per_state = self.get_valid_next_actions()

        # similar for states
        self.valid_next_states = self.get_valid_next_states()

        self.action_space = spaces.Discrete(self.A)
        # states are called observations in gym
        self.observation_space = spaces.Discrete(self.S)

        self.currentIteration = 0
        self.reset()

    def get_valid_next_actions(self) -> list:
        '''
        Pre-compute valid next actions.
        Recall that the 3-D array nextStateProbability indicates p(s'/s,a),
        and has dimension S x A x S.
        This matrix specifies that actions are invalid in a given state by
        having only zeros for a given pair (s,a). For instance, assuming S=2 states
        an A=3 actions, the matrix has an invalid action for pair (s=0, a=2):
        nextStateProbability= [[[0.1 0.9]
            [1.  0. ]
            [0.  0. ]]   <=== This indicates that action a=2 is invalid while in state 0.
            [[0.6 0.4]
            [0.  1. ]
            [1.  0. ]]]
        '''
        possible_actions_per_state = list()
        for s in range(self.S):
            possible_actions_per_state.append(list())
            for a in range(self.A):
                # check if array for pair (s,a) has only zeros:
                sum_for_s_a_pair = np.sum(self.nextStateTable[s, a])
                if sum_for_s_a_pair > 0:  # valid only if sum is larger than 0
                    possible_actions_per_state[s].append(a)
        return possible_actions_per_state

    def get_valid_next_states(self) -> list:
        '''
        Pre-compute valid next states.
        See @get_valid_next_actions
        '''
        # creates a list of lists
        valid_next_states = list()
        for s in range(self.S):
            valid_next_states.append(list())
            for a in range(self.A):
                for nexts in range(self.S):
                    p = self.nextStateTable[s, a, nexts]
                    if p != 0:
                        # here, we temporarilly allow to have duplicated entries
                        valid_next_states[s].append(nexts)
        # now we eliminate eventual duplicated entries
        for s in range(self.S):
            # convert to set
            valid_next_states[s] = set(valid_next_states[s])
            # convert back to list again
            valid_next_states[s] = list(valid_next_states[s])
        return valid_next_states

    def step(self, action: int):
        """
        The agent takes a step in the environment.
        Parameters
        ----------
        action : array of topN integers
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
            reward (float) :
            episode_over (bool) :
            info (dict) :
        """
        s = self.get_state()

        # check if the chosen action is within the set of valid actions for that state
        valid_actions = self.possible_actions_per_state[s]
        if not (action in valid_actions):
            raise Exception("Action " + str(action) +
                            " is not in valid actions list: " + str(valid_actions))

        # find next state
        nexts = np.random.choice(self.possible_states)

        # find reward value
        reward = self.rewardsTable[s, action, nexts]

        gameOver = False  # this is a continuing FMDP that never ends

        # history version with actions and states, not their indices
        # history = {"time": self.currentIteration, "state_t": self.stateListGivenIndex[s], "action_t": self.actionListGivenIndex[action],
        #           "reward_tp1": reward, "state_tp1": self.stateListGivenIndex[nexts]}
        history = {"time": self.currentIteration, "state_t": s, "action_t": action,
                   "reward_tp1": reward, "state_tp1": nexts}

        # update for next iteration
        self.currentIteration += 1  # update counter
        self.current_observation_or_state = nexts

        # state is called observation in gym API
        ob = nexts
        return ob, reward, gameOver, history

    def postprocessing_MDP_step(env, history: dict, printPostProcessingInfo: bool):
        '''This method can be overriden by subclass and process history'''
        pass  # no need to do anything here

    def get_state(self) -> int:
        """Get the current observation."""
        return self.current_observation_or_state

    def reset(self) -> int:
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        # self.currentIteration = 0
        # note there are several versions of randint!
        # self.current_observation_or_state = randint(0, self.S - 1)
        return self.nextStateTable[0]

    def get_uniform_policy_for_known_dynamics(self) -> np.ndarray:
        '''
        Takes in account the dynamics of the defined environment
        when defining actions that can be performed at each state.
        See @get_uniform_policy_for_fully_connected for
        an alternative that does not have restriction.
        '''
        policy = np.zeros((self.S, self.A))
        for s in range(self.S):
            # possible_actions_per_state is a list of lists, that indicates for each state, the list of allowed actions
            valid_actions = self.possible_actions_per_state[s]
            # no problem if denominator is zero
            uniform_probability = 1.0 / len(valid_actions)
            for a in range(len(valid_actions)):
                policy[s, a] = uniform_probability
        return policy

    def pretty_print_policy(self, policy: np.ndarray):
        '''
        Print policy.
        '''
        for s in range(self.S):
            currentState = s
            print('\ns' + str(s) + '=' + str(currentState))
            first_action = True
            for a in range(self.A):
                if policy[s, a] == 0:
                    continue
                currentAction = a
                if first_action:
                    print(' | a' + str(a) + '=' + str(currentAction), end='')
                    first_action = False  # disable this way of printing
                else:
                    print(' or a' + str(a) + '=' + str(currentAction), end='')
        print("")

if __name__ == '__main__':
    print("Main:")
    nextStateTable = np.array([[[1, 1, 0],
                                [1, 1, 0]],
                                [[0, 1, 1],
                                [0, 1, 1]],
                                [[0, 0, 1],
                                [0, 0, 1]]])
    rewardsTable = np.array([[[-3, 0, 0],
                              [-2, 5, 5]],
                             [[4, 5, 0],
                              [2, 2, 6]],
                             [[-8, 2, 80],
                              [11, 0, 3]]])

    env = GamblerRuin(nextStateTable, rewardsTable)
    print(env.reset())
    print(env.step(1))
