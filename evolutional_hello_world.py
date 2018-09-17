import sys
import random
from functools import lru_cache
from statistics import mean
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import deque


class Individual():
    '''個体
    
    ・遺伝情報
      ・長さ84のビット列で表現する
      ・それぞれ以下に発現する
        ・length(4bit)：文字列長（0~15）
        ・string(5bit * 16)：以下文字を16個
          ・スペース
          ・記号（@!,.?）
          ・小文字アルファベット（a-z）
    '''
    
    _max_len_str = 16
    _gene_size = 84
    _expression = ' !,.?@abcdefghijklmnopqrstuvwxyz'
    
    _mask_uint4 = 0b1111
    _shift_uint4 = 4
    
    _mask_char5 = 0b11111
    _shift_char5 = 5
    
    _mutate_probability = 0.05
    
    def __init__(self, target, gene=None):
        self._target = target
        
        if gene is None:
            self.gene = random.getrandbits(self._gene_size)
        else:
            assert 0 <= gene < 2 ** self._gene_size
            self.gene = gene
        
        # 遺伝情報を読み解く
        gene = self.gene
        
        info_length = (gene & self._mask_uint4) + 1
        gene = gene >> self._shift_uint4
        
        info_string = [0] * self._max_len_str
        for i in range(self._max_len_str):
            info_string[i] = (gene & self._mask_char5)
            gene = gene >> self._shift_char5
        
        # 遺伝子から個体へと発現する
        self.body = [0] * info_length
        
        for i in range(info_length):
            self.body[i] = self._expression[info_string[i]]
        
        self.body = ''.join(self.body)
        
        # 適応度を計算する
        self.fitness = lss(self.body, self._target)
    
    def _is_valid_operand(self, other):
        return hasattr(other, 'fitness')
    
    def __eq__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return self.fitness == other.fitness
    
    def __ne__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return self.fitness != other.fitness
    
    def __lt__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return self.fitness < other.fitness
    
    def __le__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return self.fitness <= other.fitness
    
    def __gt__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return self.fitness > other.fitness
    
    def __ge__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return self.fitness >= other.fitness
    
    def __hash__(self):
        return hash(self.fitness)
    
    def __repr__(self):
        return f"Individual('{self._target}', {self.gene})"
    
    def __str__(self):
        return self.body
    
    def __format__(self, format_spec):
        return self.body
    
    def __bool__(self):
        return True if self.fitness == 1.0 else False
    
    def mate(self, other, target):
        '''子供を作る'''
        assert isinstance(other, Individual)
        
        child_gene = 0
        self_gene = self.gene
        other_gene = other.gene
        
        # 両親の遺伝子を受け継ぐ
        mask_mate = random.getrandbits(self._gene_size)
        self_gene = self_gene & mask_mate
        other_gene = other_gene & ~mask_mate
        child_gene = self_gene | other_gene
        
        # 突然変異
        mask_mutation = 0
        for _ in range(self._gene_size):
            mask_mutation = mask_mutation << 1
            if random.random() <= self._mutate_probability:
                mask_mutation = mask_mutation | 0b1
        child_gene = child_gene ^ mask_mutation
        
        return Individual(target, child_gene)


class Population():
    '''集団'''
    _fertility_rate = 10
    _catastrophe_damage = 100
    
    def __init__(self, target, population_size):
        self._target = target
        self._population_size = population_size
        self.generation = [Individual(target) for _ in range(self._population_size)]
        self.generation.sort(reverse=True)
        self.generation_number = 0
    
    def next_generation(self):
        '''次世代を生み出す'''
        self.generation_number += 1
        
        # エリートと非エリートに分ける
        pareto = self._population_size // 5
        elites = self.generation[: pareto]
        non_elites = self.generation[pareto :]
        
        # エリートが非エリートと繁殖する
        children = []
        for parent1 in elites:
            while True:
                parent2 = random.choice(non_elites)
                if parent1 is parent2:
                    continue
                else:
                    break
            
            for _ in range(self._fertility_rate):
                children.append(parent1.mate(parent2, self._target))
        
        # 次世代に生存する個体を選択
        elites = random.sample(elites, 12 * len(elites) // 16)
        non_elites = random.sample(non_elites, 3 * len(non_elites) // 16)
        self.generation = elites + children + non_elites
        self.generation.sort(reverse=True)
        self.generation = self.generation[: self._population_size]
        
        # 世代の記録を残す
        max_one = self.generation[0]
        max_body = max_one.body
        max_fitness = max_one.fitness
        
        min_fitness = self.generation[-1].fitness
        mean_fitness = mean(i.fitness for i in self.generation)
        median_fitness = self.generation[self._population_size // 2].fitness
        
        return max_body, min_fitness, max_fitness, mean_fitness, median_fitness
    
    def catastrophe(self):
        '''破局の発生'''
        survivor = random.sample(self.generation, self._population_size // self._catastrophe_damage)
        newcomer = [Individual(self._target) for _ in range(self._population_size)]
        self.generation = survivor + newcomer
        self.generation.sort(reverse=True)
        self.generation = self.generation[: self._population_size]


@lru_cache(maxsize=4096)
def ld(s, t):
    '''編集距離（レーベンシュタイン距離）を計算する'''
    if not s: return len(t)
    if not t: return len(s)
    if s[0] == t[0]: return ld(s[1:], t[1:])
    l1 = ld(s, t[1:])
    l2 = ld(s[1:], t)
    l3 = ld(s[1:], t[1:])
    return 1 + min(l1, l2, l3)


def lss(s, t):
    '''類似度を計算する（編集距離を標準化し線形変換する）'''
    return -(ld(s, t) / max(len(s), len(t))) + 1


class History():
    '''履歴を管理するクラス'''
    def __init__(self):
        self.min = []
        self.max = []
        self.mean = []
        self.median = []
    
    def append(self, history):
        '''履歴を追加する'''
        assert len(history) == 4
        self.min.append(history[0])
        self.max.append(history[1])
        self.mean.append(history[2])
        self.median.append(history[3])


def visualize(title, history):
    '''履歴をグラフ化する'''
    assert isinstance(history, History)
    x = range(1, len(history.min) + 1)
    plt.figure()
    plt.plot(x, history.min, marker='.', label='min_fitness')
    plt.plot(x, history.max, marker='.', label='max_fitness')
    plt.plot(x, history.mean, marker='.', label='mean_fitness')
    plt.plot(x, history.median, marker='.', label='median_fitness')
    plt.legend(bbox_to_anchor=(0, 1), loc='upper left', fontsize=10)
    plt.grid()
    plt.xlabel('generation')
    plt.ylabel('fitness')
    plt.title(title)
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.show()


class EvolutionController():
    '''進化をつかさどる何か'''
    def __init__(self, target, population_size=1000, epochs=1000,
                 patience=60, verbose=0, log=False):
        
        self._target = target
        self._population = Population(self._target, population_size)
        self._epochs = epochs
        self._patience = patience 
        self._memory = deque([], patience)
        self._verbose = verbose
        self._log = log
        
        if self._log:
            self._history = History()
    
    def start(self):
        '''進化を開始する'''
        try:
            get_ipython()
        except:
            is_ipython = False
        else:
            is_ipython = True
        
        for i in range(1, self._epochs + 1):
            max_body, *history = self._population.next_generation()
            
            if self._verbose == 0:
                if is_ipython:
                    r = i % 4
                    
                    if r == 0:
                        s = '\r↑'
                    elif r == 1:
                        s = '\r→'
                    elif r == 2:
                        s = '\r↓'
                    else:
                        s = '\r←'
                else:
                    s = f'\033[2K\033[G{max_body}'
                
                sys.stdout.write(s)
                sys.stdout.flush()
            
            elif self._verbose > 0:
                print(f'{self._population.generation_number} : {max_body}')
            
            if self._log:
                self._history.append(history)
            
            if history[1] == 1.0:
                if self._verbose == 0:
                    if not is_ipython:
                        sys.stdout.write('\033[2K\033[G')
                        sys.stdout.flush()
                    print(f'\r{max_body}')
                
                elif self._verbose > 0:
                    print([self._population.generation[0]])
                
                if self._log:
                    visualize(max_body, self._history)
                
                break
            
            self._memory.append(history[1])
            if self._memory.count(self._memory[-1]) == self._patience:
                if self._verbose == 0:
                    if is_ipython:
                        s = '\r※'
                    else:
                        s = '\033[2K\033[GCATASTROPHE OCCURED!'
                    sys.stdout.write(s)
                    sys.stdout.flush()
                elif self._verbose > 0:
                    print('CATASTROPHE OCCURED!')
                self._population.catastrophe()
        
        else:
            if self._verbose == 0:
                if not is_ipython:
                    sys.stdout.write('\033[2K\033[G')
                    sys.stdout.flush()
                print(f'\r{max_body}')
            
            if self._log:
                visualize(max_body, self._history)


def hello_world_with_ga():
    '''遺伝的アルゴリズムでhello, world'''
    ec = EvolutionController('hello, world', epochs=1000, patience=30, verbose=1, log=True)
    ec.start()


if __name__ == '__main__':
    ec = EvolutionController('hello, world', epochs=1000, patience=30, verbose=0, log=False)
    ec.start()
