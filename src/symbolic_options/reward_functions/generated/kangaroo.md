# Answer Prompt 1

**Essential Skills for Solving Kangaroo (Atari 2600)**

---

### **1. Obstacle Avoidance**

**Description:** Dodge or punch apples and avoid contact with monkeys to survive.

**Reward Function:**

```python
def reward_obstacle_avoidance(obs: KangarooObservation, prev_obs: KangarooObservation) -> float:
    # Negative reward if player collides with any apple or monkey
    collided = any(np.array_equal(obs.player_x, x) and np.array_equal(obs.player_y, y)
                   for (x, y) in obs.apple_positions + obs.monkey_positions)
    return -1.0 if collided else 0.0
```

---

### **2. Vertical Navigation**

**Description:** Climb ladders efficiently to ascend levels and reach the child.

**Reward Function:**

```python
def reward_vertical_navigation(obs: KangarooObservation, prev_obs: KangarooObservation) -> float:
    # Reward positive change in Y (upward movement)
    dy = prev_obs.player_y - obs.player_y
    return 0.1 if dy > 0 else 0.0
```

---

### **3. Fruit Collection**

**Description:** Collect fruits and ring bell to regenerate higher-value fruits.

**Reward Function:**

```python
def reward_fruit_collection(obs: KangarooObservation, prev_obs: KangarooObservation) -> float:
    # Reward for reducing fruit count, indicating a fruit was picked
    reward = 0.0
    if len(obs.fruit_positions) < len(prev_obs.fruit_positions):
        reward += 0.5
    # Extra reward for reaching bell position to regenerate fruit
    if np.array_equal(obs.player_x, obs.bell_position[0]) and np.array_equal(obs.player_y, obs.bell_position[1]):
        reward += 0.2
    return reward
```

---

### **4. Goal Reaching (Child Rescue)**

**Description:** Navigate to child’s position to complete a level.

**Reward Function:**

```python
def reward_reach_child(obs: KangarooObservation, prev_obs: KangarooObservation) -> float:
    # High reward if player reaches the child
    return 1.0 if np.array_equal(obs.player_x, obs.child_position[0]) and np.array_equal(obs.player_y, obs.child_position[1]) else 0.0
```

---

These modular skills can be combined in a hierarchical RL setup with high-level control choosing which skill to execute based on context.







----------------------------------------------
# Answer Prompt 2


To select skills effectively, we prioritize **safety first (Obstacle Avoidance)**, then **progress (Vertical Navigation / Goal Reaching)**, and **fruit collection** when safe and beneficial.

---

### **Skill Selection Strategy**

At each time step, the skill selector operates by the following priority:

1. **Obstacle Avoidance:** If any apple or monkey is within a danger zone (close proximity).
2. **Goal Reaching:** If the player is on the highest reachable platform and near the child.
3. **Fruit Collection:** If fruit is present and reachable.
4. **Vertical Navigation:** Default skill to ascend and make progress.

---

### **Skill Selector Function**

```python
def select_skill(obs: KangarooObservation, prev_obs: KangarooObservation) -> str:
    player_pos = np.array([obs.player_x, obs.player_y])
    
    # Danger zone: apples or monkeys within a small radius
    def in_danger():
        danger_entities = obs.apple_positions + obs.monkey_positions
        for (x, y) in danger_entities:
            if np.linalg.norm(player_pos - np.array([x, y])) <= 1.0:
                return True
        return False

    # Near child and roughly at the same height
    def near_child():
        child_pos = np.array([obs.child_position[0], obs.child_position[1]])
        return np.linalg.norm(player_pos - child_pos) <= 1.5

    # Any reachable fruit within vertical +/-1 and some horizontal range
    def fruit_visible():
        for (x, y) in obs.fruit_positions:
            if abs(y - obs.player_y) <= 1 and abs(x - obs.player_x) <= 3:
                return True
        return False

    if in_danger():
        return "obstacle_avoidance"
    elif near_child():
        return "goal_reaching"
    elif fruit_visible():
        return "fruit_collection"
    else:
        return "vertical_navigation"
```

---

This decision policy enables a hierarchical controller to sequence the primitive skills safely and efficiently for high reward acquisition.
