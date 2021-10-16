from gekko import GEKKO
import numpy as np

def gekko():

    m = GEKKO() # Initialize gekko
    m.options.SOLVER = 1  # APOPT is an MINLP solver

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
    use_numpy = True
    use_integer = True
    
    if use_numpy:
        x1 = np.array(m.Var(value=1,lb=1,ub=5))
        x2 = np.array(m.Var(value=5,lb=1,ub=5))
        # Integer constraints for x3 and x4
        x3 = np.array(m.Var(value=5,lb=1,ub=5,integer=use_integer))
        x4 = np.array(m.Var(value=1,lb=1,ub=5,integer=use_integer))
        
        
    else:
        x1 = m.Var(value=1,lb=1,ub=5)
        x2 = m.Var(value=5,lb=1,ub=5)
        # Integer constraints for x3 and x4
        x3 = m.Var(value=5,lb=1,ub=5,integer=use_integer)
        x4 = m.Var(value=1,lb=1,ub=5,integer=use_integer)
    

    # Equations
    
    m.Equation(x1*x2*x3*x4>=25)
    m.Equation(x1**2+x2**2+x3**2+x4**2==40)

    # Objective function
    
    m.Obj(x1*x4*(x1+x2+x3)+x3) # Objective

    # Solve
    
    m.solve(disp=False) # Solve
    print('Results')
    
    # Print Results
    
    if use_numpy: # if array is type numpy
        
        print("Using numpy:")
        print('x1: ' + np.array2string(x1))
        print('x2: ' + np.array2string(x2))
        print('x3: ' + np.array2string(x3))
        print('x4: ' + np.array2string(x4))
        
    else: # if array is list
        print("Using list:")
        print('x1: ' + str(x1.value))
        print('x2: ' + str(x2.value))
        print('x3: ' + str(x3.value))
        print('x4: ' + str(x4.value))
    
    print('Objective: ' + str(m.options.objfcnval))


def gekko2(): # Test multiplication with numpy

    m = GEKKO() # Initialize gekko
    m.options.SOLVER = 1  # APOPT is an MINLP solver

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
    use_numpy = False
    use_integer = True
    
    if use_numpy:
        x1 = np.array(m.Var(value=1,lb=1,ub=5))
        x2 = np.array(m.Var(value=5,lb=1,ub=5))
        # Integer constraints for x3 and x4
        x3 = np.array(m.Var(value=5,lb=1,ub=5,integer=use_integer))
        x4 = np.array(m.Var(value=1,lb=1,ub=5,integer=use_integer))
        
        
    else:
        x1 = m.Var(value=1,lb=1,ub=5)
        x2 = m.Var(value=5,lb=1,ub=5)
        # Integer constraints for x3 and x4
        x3 = m.Var(value=5,lb=1,ub=5,integer=use_integer)
        x4 = m.Var(value=1,lb=1,ub=5,integer=use_integer)
    

    # Equations
    
    m.Equation(x1*x2*x3*x4>=25)
    m.Equation(x1**2+x2**2+x3**2+x4**2==40)

    # Objective function
    
    m.Obj(x1*x4*(x1+x2+x3)+x3) # Objective

    # Solve
    
    m.solve(disp=False) # Solve
    print('Results')
    
    # Print Results
    
    if use_numpy: # if array is type numpy
        
        print("Using numpy:")
        print('x1: ' + np.array2string(x1))
        print('x2: ' + np.array2string(x2))
        print('x3: ' + np.array2string(x3))
        print('x4: ' + np.array2string(x4))
        
    else: # if array is list
        print("Using list:")
        print('x1: ' + str(x1.value))
        print('x2: ' + str(x2.value))
        print('x3: ' + str(x3.value))
        print('x4: ' + str(x4.value))
    
    print('Objective: ' + str(m.options.objfcnval))

def main():

    gekko2()

if __name__ == '__main__':

    main()
