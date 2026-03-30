import math

class RewardFunction:
    def __init__(self):
        pass

    # To implement...
    def get_reward(self, s_info):
        raise NotImplementedError("To be implemented")

    def get_type(self):
        raise NotImplementedError("To be implemented")


class ConstantRewardFunction(RewardFunction):
    """
    Defines a constant reward for a 'simple reward machine'
    """
    def __init__(self, c):
        super().__init__()
        self.c = c

    def get_type(self):
        return "constant"

    def get_reward(self, s_info):
        return self.c

class RewardControl(RewardFunction):
    """
    Gives a reward for moving forward
    """
    def __init__(self):
        super().__init__()

    def get_type(self):
        return "ctrl"

    def get_reward(self, s_info):
        return s_info['reward_ctrl']

class RewardForward(RewardFunction):
    """
    Gives a reward for moving forward
    """
    def __init__(self):
        super().__init__()

    def get_type(self):
        return "forward"

    def get_reward(self, s_info):
        return s_info['reward_run'] + s_info['reward_ctrl']  #Cheetah


class RewardBackwards(RewardFunction):
    """
    Gives a reward for moving backwards
    """
    def __init__(self):
        super().__init__()

    def get_type(self):
        return "backwards"

    def get_reward(self, s_info):
        return -s_info['reward_run'] + s_info['reward_ctrl']  #Cheetah


## Reward functions for Seaquest
class RewardCollectDivers(RewardFunction):
    """Reward function for State 0: Focuses on picking up divers and surviving."""
    def __init__(self):
        super().__init__()

    def get_type(self):
        return "collect_divers"

    def get_reward(self, s_info):
        reward = 0.0
        # Reward for picking up a diver
        if s_info['collected_divers'] > s_info['prev_collected_divers']:
            reward += 50.0
        # Reward for shooting enemies (indicated by score increase without surfacing)
        if s_info['player_score'] > s_info['prev_player_score']:
            reward += 5.0
        # Penalty for losing a life
        if s_info['lives'] < s_info['prev_lives']:
            reward -= 100.0
        return reward

class RewardSurfacing(RewardFunction):
    """Reward function for State 1: Focuses on reaching the surface quickly."""
    def __init__(self):
        super().__init__()

    def get_type(self):
        return "surfacing"

    def get_reward(self, s_info):
        reward = 0.0
        # Dense reward for moving upwards (assuming lower y is higher up on screen)
        if s_info['player_y'] < s_info['prev_player_y']:
            reward += 1.0
        # Large reward for successfully surfacing and dropping off divers
        if s_info['collected_divers'] < s_info['prev_collected_divers'] and s_info['player_y'] <= 10:
            reward += 200.0
        # Penalty for losing a life (e.g., hitting the surface patrol sub or running out of oxygen)
        if s_info['lives'] < s_info['prev_lives']:
            reward -= 100.0
        return reward

class RewardTerminal(RewardFunction):
    """Reward function for State 2: Game over."""
    def __init__(self):
        super().__init__()

    def get_type(self):
        return "terminal"

    def get_reward(self, s_info):
        return 0.0

## Reward functions for Kangaroo
class RewardNavigation(RewardFunction):
    """Reward for reducing Manhattan distance between player and child."""

    def __init__(self):
        super().__init__()

    def get_type(self):
        return "navigation"

    def get_reward(self, s_info):
        prev_dist = abs(s_info["prev_player_x"] - s_info["prev_child_x"]) + abs(
            s_info["prev_player_y"] - s_info["prev_child_y"]
        )
        curr_dist = abs(s_info["player_x"] - s_info["child_x"]) + abs(
            s_info["player_y"] - s_info["child_y"]
        )
        return 1.0 if curr_dist < prev_dist else -0.1


class RewardCombat(RewardFunction):
    """Reward for reducing the number of active monkeys."""

    def __init__(self):
        super().__init__()

    def get_type(self):
        return "combat"

    def get_reward(self, s_info):
        if s_info["active_monkeys"] < s_info["prev_active_monkeys"]:
            return 10.0
        return 0.0


class RewardCollection(RewardFunction):
    """Reward for reducing the number of active fruits (fruit picked up)."""

    def __init__(self):
        super().__init__()

    def get_type(self):
        return "collection"

    def get_reward(self, s_info):
        return 5.0 if s_info["active_fruits"] < s_info["prev_active_fruits"] else 0.0


class RewardSurvival(RewardFunction):
    """Penalty on life loss plus a small heartbeat reward while alive."""

    def __init__(self):
        super().__init__()

    def get_type(self):
        return "survival"

    def get_reward(self, s_info):
        if s_info["lives"] < s_info["prev_lives"]:
            return -50.0
        return 0.1