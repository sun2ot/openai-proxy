import numpy as np

class HiddenMarkov:
    def __init__(self):
        self.alphas = None
        self.forward_P = None
        self.betas = None
        self.backward_P = None

    # 前向算法
    def forward(self, Q, V, A, B, O, PI):
        """
        Q: 所有可能状态值集合
        V: 所有可能观测值集合
        A: 状态转移概率矩阵
        B: 观测概率矩阵
        O: 观测序列
        PI: 初始概率分布向量
        """
        # 状态序列的大小
        N = len(Q)
        # 观测序列的大小
        M = len(O)
        # 初始化前向概率alpha值
        alphas = np.zeros((N, M))
        # 时刻数=观测序列数
        T = M
        # 遍历每一个时刻，计算前向概率alpha值
        for t in range(T):
            # 观测序列对应的索引
            indexOfO = V.index(O[t])
            # 遍历状态序列
            for i in range(N):
                # 初始化alpha初值
                if t == 0:
                    # 注意, 这里alpha的值是按列存的
                    alphas[i][t] = PI[t][i] * B[i][indexOfO]
                    print('alpha_{i}({t})={p:.4f}'.format(i=i+1,t=t+1,p=alphas[i][t]))
                else:
                    # "逐行取某一列的值"，得到矩阵的一列
                    alphas[i][t] = np.dot([alpha[t - 1] for alpha in alphas],
                                          [a[i] for a in A]) * B[i][indexOfO]
                    print('alpha_{i}({t})={p:.4f}'.format(i=i + 1, t=t + 1, p=alphas[i][t]))
        # T时刻，N个alpha_i(T)的值求和
        self.forward_P = np.sum([alpha[T - 1] for alpha in alphas])
        print('p(o|\lambda)={}'.format(self.forward_P))
        self.alphas = alphas



    # 后向算法
    def backward(self, Q, V, A, B, O, PI):
        # 状态序列的大小
        N = len(Q)
        # 观测序列的大小
        T = len(O)
        # 初始化后向概率beta值
        betas = np.ones((N, T))
        # 初始化 \beta_i(T)=1
        for i in range(N):
            print('beta_{i}({T})=1'.format(i=i+1, T=T))
        # 对观测序列逆向遍历(索引为T-2到0)
        for t in range(T - 2, -1, -1):
            # 得到序列对应的索引
            indexOfO = V.index(O[t + 1])
            # 遍历状态序列
            for i in range(N):
                betas[i][t] = np.dot(
                    np.dot(A[i], [b[indexOfO] for b in B]),
                    [beta[t + 1] for beta in betas])
                print('beta_{i}({t})={r:.4f}'.format(i=i+1,t=t+1,r=betas[i][t]))
        # 取出第一个值
        indexOfO = V.index(O[0])
        self.betas = betas

        P = np.dot(np.multiply(PI, [b[indexOfO] for b in B]),
                   [beta[0] for beta in betas])
        self.backward_P = P
        print("P(O|lambda) = {}".format(self.backward_P))


    # 维特比算法
    def viterbi(self, Q, V, A, B, O, PI):
        # 状态序列的大小
        N = len(Q)
        # 观测序列的大小
        M = len(O)
        # 初始化daltas
        deltas = np.zeros((N, M))
        # 初始化psis
        psis = np.zeros((N, M))
        # 初始化最优路径矩阵，该矩阵维度与观测序列维度相同
        I = np.zeros((1, M))
        # 遍历观测序列
        for t in range(M):
            # 递推从t=2开始
            realT = t + 1
            # 得到序列对应的索引
            indexOfO = V.index(O[t])
            for i in range(N):
                realI = i + 1
                if t == 0:
                    # P185 算法10.5 步骤(1)
                    deltas[i][t] = PI[0][i] * B[i][indexOfO]
                    psis[i][t] = 0
                    print('delta1(%d) = pi%d * b%d(o1) = %.2f * %.2f = %.2f' %
                          (realI, realI, realI, PI[0][i], B[i][indexOfO],
                           deltas[i][t]))
                    print('psis1(%d) = 0' % (realI))
                else:
                    # # P185 算法10.5 步骤(2)
                    deltas[i][t] = np.max(
                        np.multiply([delta[t - 1] for delta in deltas],
                                    [a[i] for a in A])) * B[i][indexOfO]
                    print(
                        'delta%d(%d) = max[delta%d(j)aj%d]b%d(o%d) = %.2f * %.2f = %.5f'
                        % (realT, realI, realT - 1, realI, realI, realT,
                           np.max(
                               np.multiply([delta[t - 1] for delta in deltas],
                                           [a[i] for a in A])), B[i][indexOfO],
                           deltas[i][t]))
                    psis[i][t] = np.argmax(
                        np.multiply([delta[t - 1] for delta in deltas],
                                    [a[i] for a in A]))
                    print('psis%d(%d) = argmax[delta%d(j)aj%d] = %d' %
                          (realT, realI, realT - 1, realI, psis[i][t]))
        #print(deltas)
        #print(psis)
        # 得到最优路径的终结点
        I[0][M - 1] = np.argmax([delta[M - 1] for delta in deltas])
        print('i%d = argmax[deltaT(i)] = %d' % (M, I[0][M - 1] + 1))
        # 递归由后向前得到其他结点
        for t in range(M - 2, -1, -1):
            I[0][t] = psis[int(I[0][t + 1])][t + 1]
            print('i%d = psis%d(i%d) = %d' %
                  (t + 1, t + 2, t + 2, I[0][t] + 1))
        # 输出最优路径
        print('最优路径是：', "->".join([str(int(i + 1)) for i in I[0]]))



# 测试
Q = [1, 2, 3]
V = ['红', '白']
A = [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
# O = ['红', '白', '红', '红', '白', '红', '白', '白']
O = ['红', '白', '红']    #习题10.1的例子
PI = [[0.2, 0.4, 0.4]]

HMM = HiddenMarkov()
HMM.forward(Q, V, A, B, O, PI)
# HMM.backward(Q, V, A, B, O, PI)
# HMM.viterbi(Q, V, A, B, O, PI)