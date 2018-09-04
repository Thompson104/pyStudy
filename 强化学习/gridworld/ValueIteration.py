import numpy as np
from gridworld import GridworldEnv

env = GridworldEnv()
# 通过迭代更新v表格，计算出每个状态的期望价值
def value_iteration(env,theta=0.001,discount_factor=1.0):
    # state当前状态，v为所有状态的value
    def one_step_lookahead(state,v):
        '''
        计算状态state向各个方向移动后的value，目的是找出最好的方向
        :param state: 当前状态
        :param v:     当前各状态的value表
        :return:      返回一个list，包含各个action后对应的期望value
        '''
        A = np.zeros(env.nA)
        # 计算每个方向
        for a in range(env.nA):
            #env.P[state][a]当前状态的动作a下的转移概率
            for prob,next_state,reward,done in env.P[state][a]:
                # 计算a动作后的value
                A[a] +=  prob * (reward + discount_factor * v[next_state])
        return A

    #初始化每个状态的value为零
    v = np.zeros(env.nS)

    # 迭代更新v表
    while True:
        print('=============')
        delta = 0
        # 每个状态进行迭代,更新相应的最佳动作
        for s in range(env.nS):
            # 计算状态s进行4个方向动作之后的value
            A = one_step_lookahead(s,v)
            best_action_value = np.max(A)
            delta = max(delta,np.abs(best_action_value - v[s]))
            v[s] = best_action_value
        if delta < theta:
            break

    # 每个状态对应的4个矩阵
    policy = np.zeros((env.nS,env.nA))

    for s in range(env.nS):
        A = one_step_lookahead(s,v)
        best_action = np.argmax(A)
        policy[s,best_action] = 1.0
    return policy,v,

policy,v = value_iteration(env)

print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")
