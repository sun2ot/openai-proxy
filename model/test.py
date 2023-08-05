# 定义转移概率矩阵和发射概率矩阵
transition_probs = {
    'X': {'X': 0.7, 'Y': 0.3},
    'Y': {'X': 0.4, 'Y': 0.6}
}

emission_probs = {
    'X': {'A': 0.3, 'B': 0.7},
    'Y': {'A': 0.6, 'B': 0.4}
}

# 定义观测序列
observations = ['A', 'B', 'A']

# 初始化前向概率矩阵
forward_prob = [{state: 0.0 for state in transition_probs} for _ in range(len(observations))]
forward_prob[0] = {state: emission_probs[state][observations[0]] for state in transition_probs}

# 递推计算前向概率
for t in range(1, len(observations)):
    for current_state in transition_probs:
        forward_prob[t][current_state] = max(
            forward_prob[t-1][prev_state] * transition_probs[prev_state][current_state]
            * emission_probs[current_state][observations[t]]
            for prev_state in transition_probs
        )

# 找到最优路径
best_path = []
max_prob = 0.0
for t in range(len(observations)-1, -1, -1):
    if t == len(observations) - 1:
        best_state = max(transition_probs, key=lambda state: forward_prob[t][state])
        best_path.append(best_state)
        max_prob = forward_prob[t][best_state]
    else:
        prev_state = best_path[-1]
        best_state = max(transition_probs, key=lambda state: forward_prob[t][state] * transition_probs[state][prev_state])
        best_path.append(best_state)

best_path.reverse()

# 打印最优路径和概率
print("Best path:", best_path)
print("Probability:", max_prob)
