# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 12:30:24 2021

@author: James Ang
"""

import pulp
import time
from gekko import GEKKO

# from numpy.random import rand
from scipy.optimize import minimize, brute


def timer(func):
    def wrapper():
        before = time.time()
        func()
        print("Function takes",time.time()-before,"seconds")

    return wrapper


def objfunq1(x):
    print(x[0], x[1])
    
    return (x[0] - 1)**2 + (x[1] - 2.5)**2

def opt1():
    cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},
            {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
            {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})
    
    bnds = ((0, None), (0, None))
    
    res = minimize(objfunq1, (2, 0), method='SLSQP', bounds=bnds,
                   constraints=cons)
    
    return res


def get_data():
    
    cost = [ 100, 140, 150, 105, 240]
    target = 3
    truck_mt = [ 2, 3, 3, 2, 5]
    # truck_mt = [i for i in truck_mt]
    
    return cost, target, truck_mt

def objfunq2(x):
    
    print(x)

    cost = get_data()[0]
    
    return sum([i*j for i,j in zip(x,cost)]) #x[0]*100 + x[1]*200 + x[2]*150

def opt2():
    
    
    target, truck_mt = get_data() [1:]

    cons = [
            {'type':'ineq', 'fun': lambda x, target = target, truck_mt = truck_mt: sum([i*j for i,j in zip(x,truck_mt)]) - target},
            {'type':'ineq', 'fun': lambda x, target = target, truck_mt = truck_mt: -(sum([i*j for i,j in zip(x,truck_mt)]) - target)},
            {'type':'eq','fun': lambda x : max([x[i]-int(x[i]) for i in range(len(x))])},
            # {'type':'ineq','fun': lambda x : -(max([x[i]-int(x[i]) for i in range(len(x))]))}
            ]
    
    t = (0,1)*len(truck_mt)
        
    bnds = tuple([(t[i], t[i+1]) for i in range(0,len(t),2)])
    
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Setting Decision Variables Boundaries as Inequalities
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""
    for factor in range(len(bnds)):
        # print(factor)
        lower, upper = bnds[factor]
        
        l = {'type': 'ineq',
             'fun': lambda x, lb=lower, i=factor: x[i] - lb}
        u = {'type': 'ineq',
             'fun': lambda x, ub=upper, i=factor: ub - x[i]}
    
    cons.append(l)
    cons.append(u)
    
    # # define range for input
    # r_min, r_max = 0, 1.0
    # # define the starting point as a random sample from the domain
    # pt = r_min + rand(1) * (r_max - r_min)
    
    res = minimize(objfunq2, 
                   (1,)*len(truck_mt),
                   # pt,
                   # method='COBYLA',
                   method='SLSQP',
                    # jac=True,
                    # hess='cs', 
                   bounds=bnds,
                   constraints=cons,
                   # tol=1e-6, 
                   # options={'rhobeg': 0.3}
                   )
    
    return res

def test_sol():
    
    data = [(481.79, 5), (412.04, 4), (365.54, 3)] #, (375.88, 3), (379.75, 3), (632.92, 5), (127.89, 1), (835.71, 6), (200.21, 1)]
    x = pulp.LpVariable.dicts('x', range(len(data)), lowBound=0, upBound=7, cat=pulp.LpInteger)
    
    numerator = dict((i,tup[0]) for i,tup in enumerate(data))
    denom_int = dict((i,tup[1]) for i,tup in enumerate(data))
    
    problem = pulp.LpProblem('Mixed Integer Linear Programming', sense=pulp.LpMinimize)
    
    # objective function (doesn't work)
    # TypeError: unsupported operand type(s) for /: 'float' and 'LpAffineExpression'
    problem += sum([numerator[i] / (denom_int[i] + x[i]) for i in range(len(data))])
    
    problem.solve()
    
    for v in problem.variables():
      print(v.name, "=", v.varValue)

def test_brute():
    def f(x):
        return (481.79/(5+x[0]))+(412.04/(4+x[1]))+(365.54/(3+x[2]))
    
    ranges = (slice(0, 1, 1),) * 4
    result = brute(objfunq2, ranges, disp=True, finish=None)
    print([round(i) for i in result.x])

def test_gekko():
    m = GEKKO()
    x,y = m.Array(m.Var,2,integer=True,lb=0)
    m.Maximize(y)
    m.Equations([-x+y<=1,
                 3*x+2*y<=12,
                 2*x+3*y<=12])
    m.options.SOLVER = 1
    m.solve()
    print('Objective: ', -m.options.OBJFCNVAL)
    print('x: ', x.value[0])
    print('y: ', y.value[0])

@timer
def gekko2():
    
    cost, target, truck_mt = get_data()
    
    m = GEKKO() # Initialize gekko
    m.options.SOLVER=1  # APOPT is an MINLP solver
    
    # optional solver settings with APOPT
    m.solver_options = ['minlp_maximum_iterations 500', \
                        # minlp iterations with integer solution
                        'minlp_max_iter_with_int_sol 10', \
                        # treat minlp as nlp
                        'minlp_as_nlp 0', \
                        # nlp sub-problem max iterations
                        'nlp_maximum_iterations 50', \
                        # 1 = depth first, 2 = breadth first
                        'minlp_branch_method 1', \
                        # maximum deviation from whole number
                        'minlp_integer_tol 0.05', \
                        # covergence tolerance
                        'minlp_gap_tol 0.01']
    
    # Initialize variables
    x = [m.Var(value=1,lb=0,ub=1,integer=True) for _ in range(len(truck_mt))]
    
    # x1 = m.Var(value=1,lb=0,ub=1,integer=True)
    # x2 = m.Var(value=1,lb=0,ub=1,integer=True)
    
    # # Integer constraints for x3 and x4
    # x3 = m.Var(value=1,lb=0,ub=1,integer=True)
    # x4 = m.Var(value=1,lb=0,ub=1,integer=True)
    # x5 = m.Var(value=1,lb=0,ub=1,integer=True)
    
    # Equations
    m.Equation(sum([i*j for i,j in zip(x,truck_mt)])>=target)
    # m.Equation(2*x[0] + 3*x[1]>=target)
    # m.Equation(2*x1 + 3*x2 + 3*x3 + 2*x4 +5*x5==target)
    m.Obj(sum([i*j for i,j in zip(x,cost)]))
    # m.Obj(100*x[0] + 140*x[1])
    # m.Obj(100*x1 + 140*x2 + 150*x3 + 105*x4 + 240*x5) # Objective
    m.solve(disp=False) # Solve
    print('Results')
    print('x: ' + str([i.value for i in x]))
    print('x1: ' + str(x[0].value))
    print('x2: ' + str(x[1].value))
    # print('x3: ' + str(x3.value))
    # print('x4: ' + str(x4.value))
    # print('x5: ' + str(x5.value))
    print('Objective: ' + str(m.options.objfcnval))

def main():
    # res1 = opt1()
    # print(res1)
    
    # res2 = opt2()
    # print(res2)
    # print([round(i) for i in res2.x])
    
    # test_sol()
    # test_brute()
    # test_gekko()
    gekko2()
    
if __name__ == '__main__':
    
    main()