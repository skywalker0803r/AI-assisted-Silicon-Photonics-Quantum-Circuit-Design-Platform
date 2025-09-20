"""
Genetic Algorithm for silicon photonics design optimization
遺傳演算法最佳化模組
"""
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
import random
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class Individual:
    """個體類別"""
    genes: Dict[str, float]
    fitness: Optional[float] = None
    objectives: Optional[List[float]] = None
    
class GeneticOptimizer:
    """遺傳演算法最佳化器"""
    
    def __init__(self, bounds: Dict[str, Tuple[float, float]], 
                 population_size: int = 50,
                 n_generations: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 multi_objective: bool = False,
                 random_seed: Optional[int] = None):
        """
        初始化遺傳演算法
        
        Args:
            bounds: 參數邊界 {'param_name': (min, max)}
            population_size: 族群大小
            n_generations: 世代數
            crossover_rate: 交配率
            mutation_rate: 突變率
            multi_objective: 是否為多目標最佳化
            random_seed: 隨機種子
        """
        self.bounds = bounds
        self.param_names = list(bounds.keys())
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.multi_objective = multi_objective
        
        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        self.population = []
        self.best_individual = None
        self.evolution_history = []
        
    def _create_random_individual(self) -> Individual:
        """創建隨機個體"""
        genes = {}
        for param_name in self.param_names:
            min_val, max_val = self.bounds[param_name]
            genes[param_name] = random.uniform(min_val, max_val)
        return Individual(genes=genes)
    
    def _initialize_population(self):
        """初始化族群"""
        self.population = []
        for _ in range(self.population_size):
            individual = self._create_random_individual()
            self.population.append(individual)
    
    def _evaluate_population(self, objective_func: Callable):
        """評估族群適應度"""
        for individual in self.population:
            if individual.fitness is None:
                if self.multi_objective:
                    objectives = objective_func(individual.genes)
                    individual.objectives = objectives
                    # 多目標情況下，使用支配排序
                    individual.fitness = self._calculate_dominance_rank(objectives)
                else:
                    fitness = objective_func(individual.genes)
                    individual.fitness = fitness
    
    def _calculate_dominance_rank(self, objectives: List[float]) -> float:
        """計算支配排序（簡化版）"""
        # 這裡使用簡化的加權和作為fitness，實際應該用NSGA-II等算法
        return sum(objectives) / len(objectives)
    
    def _tournament_selection(self, tournament_size: int = 3) -> Individual:
        """錦標賽選擇"""
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """交配操作（算術交配）"""
        alpha = random.random()  # 交配係數
        
        child1_genes = {}
        child2_genes = {}
        
        for param_name in self.param_names:
            p1_val = parent1.genes[param_name]
            p2_val = parent2.genes[param_name]
            
            child1_genes[param_name] = alpha * p1_val + (1 - alpha) * p2_val
            child2_genes[param_name] = (1 - alpha) * p1_val + alpha * p2_val
            
            # 確保在邊界內
            min_val, max_val = self.bounds[param_name]
            child1_genes[param_name] = np.clip(child1_genes[param_name], min_val, max_val)
            child2_genes[param_name] = np.clip(child2_genes[param_name], min_val, max_val)
        
        return Individual(genes=child1_genes), Individual(genes=child2_genes)
    
    def _mutate(self, individual: Individual) -> Individual:
        """突變操作（高斯突變）"""
        mutated_genes = individual.genes.copy()
        
        for param_name in self.param_names:
            if random.random() < self.mutation_rate:
                min_val, max_val = self.bounds[param_name]
                range_val = max_val - min_val
                
                # 高斯突變
                mutation_strength = 0.1 * range_val
                current_val = mutated_genes[param_name]
                mutation = np.random.normal(0, mutation_strength)
                
                new_val = current_val + mutation
                mutated_genes[param_name] = np.clip(new_val, min_val, max_val)
        
        return Individual(genes=mutated_genes)
    
    def _selection_and_reproduction(self) -> List[Individual]:
        """選擇和繁殖"""
        new_population = []
        
        # 保留最佳個體（精英策略）
        elite_size = max(1, self.population_size // 10)
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        new_population.extend(sorted_pop[:elite_size])
        
        # 生成新個體
        while len(new_population) < self.population_size:
            # 選擇父母
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # 交配
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            
            # 突變
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            
            new_population.extend([child1, child2])
        
        # 確保族群大小正確
        return new_population[:self.population_size]
    
    def optimize(self, objective_func: Callable, verbose: bool = True) -> List[Individual]:
        """執行遺傳演算法最佳化"""
        # 初始化
        self._initialize_population()
        
        for generation in range(self.n_generations):
            # 評估適應度
            self._evaluate_population(objective_func)
            
            # 記錄歷史
            current_best = max(self.population, key=lambda x: x.fitness)
            if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
                self.best_individual = current_best
            
            # 記錄統計資訊
            fitnesses = [ind.fitness for ind in self.population]
            self.evolution_history.append({
                'generation': generation,
                'best_fitness': max(fitnesses),
                'avg_fitness': np.mean(fitnesses),
                'std_fitness': np.std(fitnesses)
            })
            
            if verbose and (generation + 1) % 10 == 0:
                print(f"世代 {generation + 1}/{self.n_generations}, "
                      f"最佳適應度: {max(fitnesses):.4f}, "
                      f"平均適應度: {np.mean(fitnesses):.4f}")
            
            # 選擇和繁殖
            if generation < self.n_generations - 1:  # 最後一代不需要繁殖
                self.population = self._selection_and_reproduction()
        
        # 最終評估
        self._evaluate_population(objective_func)
        
        if verbose:
            final_best = max(self.population, key=lambda x: x.fitness)
            print(f"\n=== 遺傳演算法完成 ===")
            print(f"最佳適應度: {final_best.fitness:.4f}")
            print("最佳參數:")
            for param, value in final_best.genes.items():
                print(f"  {param}: {value:.4f}")
        
        # 返回最佳解或Pareto前緣
        if self.multi_objective:
            return self._get_pareto_front()
        else:
            return [max(self.population, key=lambda x: x.fitness)]
    
    def _get_pareto_front(self) -> List[Individual]:
        """獲取Pareto前緣"""
        if not self.multi_objective:
            return [self.best_individual]
        
        pareto_front = []
        
        for individual in self.population:
            is_dominated = False
            
            for other in self.population:
                if individual != other and self._dominates(other.objectives, individual.objectives):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(individual)
        
        return pareto_front
    
    def _dominates(self, obj1: List[float], obj2: List[float]) -> bool:
        """檢查obj1是否支配obj2"""
        better_in_any = False
        for o1, o2 in zip(obj1, obj2):
            if o1 < o2:
                return False
            elif o1 > o2:
                better_in_any = True
        return better_in_any
    
    def plot_evolution_history(self):
        """繪製演化歷史"""
        if not self.evolution_history:
            print("沒有演化歷史數據！")
            return
        
        generations = [entry['generation'] for entry in self.evolution_history]
        best_fitnesses = [entry['best_fitness'] for entry in self.evolution_history]
        avg_fitnesses = [entry['avg_fitness'] for entry in self.evolution_history]
        std_fitnesses = [entry['std_fitness'] for entry in self.evolution_history]
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(generations, best_fitnesses, 'r-', linewidth=2, label='最佳適應度')
        plt.plot(generations, avg_fitnesses, 'b-', label='平均適應度')
        plt.fill_between(generations, 
                        np.array(avg_fitnesses) - np.array(std_fitnesses),
                        np.array(avg_fitnesses) + np.array(std_fitnesses),
                        alpha=0.3, color='blue')
        plt.xlabel('世代')
        plt.ylabel('適應度')
        plt.title('遺傳演算法收斂歷史')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(generations, std_fitnesses, 'g-', label='適應度標準差')
        plt.xlabel('世代')
        plt.ylabel('族群多樣性')
        plt.title('族群多樣性變化')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# 便利函數
def genetic_optimize(objective_func: Callable,
                    bounds: Dict[str, Tuple[float, float]],
                    population_size: int = 50,
                    n_generations: int = 100,
                    multi_objective: bool = False,
                    verbose: bool = True) -> Tuple[Dict[str, float], float]:
    """
    便利的遺傳演算法最佳化函數
    
    Returns:
        最佳參數和最佳適應度
    """
    optimizer = GeneticOptimizer(
        bounds=bounds,
        population_size=population_size,
        n_generations=n_generations,
        multi_objective=multi_objective
    )
    
    best_solutions = optimizer.optimize(objective_func, verbose=verbose)
    
    if multi_objective:
        # 多目標情況下返回第一個Pareto解
        if best_solutions:
            return best_solutions[0].genes, best_solutions[0].fitness
        else:
            return {}, 0.0
    else:
        best_solution = best_solutions[0]
        return best_solution.genes, best_solution.fitness

# 測試範例
if __name__ == "__main__":
    # 測試函數：最大化 Himmelblau 函數的負值
    def himmelblau(params):
        x, y = params['x'], params['y']
        return -((x**2 + y - 11)**2 + (x + y**2 - 7)**2)
    
    bounds = {
        'x': (-5.0, 5.0),
        'y': (-5.0, 5.0)
    }
    
    print("測試遺傳演算法...")
    best_params, best_fitness = genetic_optimize(
        himmelblau, bounds, 
        population_size=30, 
        n_generations=50,
        verbose=True
    )
    
    print(f"\nHimmelblau函數有4個全域最佳解:")
    print(f"找到的解: x={best_params['x']:.3f}, y={best_params['y']:.3f}")
    print(f"函數值: {best_fitness:.3f}")