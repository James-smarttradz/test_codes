# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 16:14:39 2021

@author: James Ang
"""
import time
import json
import math
import pandas as pd
from datetime import datetime
from gekko import GEKKO
import numpy as np

def get_data():

    f = open('input_huge2.json', 'r')
    data = json.load(f)
    f.close()
    
    f = open('truckAvailability.json', 'r')
    data_truck = json.load(f)
    f.close()
    
    aggre_demand = data.get('aggregated_demand')
    
    selectedDeliveryDate = data.get('deliveryDate')
    
    return aggre_demand, selectedDeliveryDate, data_truck

def get_data1():
    
    rentalRate = [ 100, 140, 150, 105, 240]
    target = 3
    truck_size_list = [ 2, 3, 3, 2, 5]
    # truck_size_list = [i for i in truck_size_list]
    
    return rentalRate, target, truck_size_list

def timer(func):
    def wrapper(target, rentalRate, truck_size_list):
        before = time.time()
        func(target, rentalRate, truck_size_list)
        print("Function takes",time.time()-before,"seconds")
        print("=================================================\n\n")

    return wrapper

def roundup(x):
    
    return int(math.ceil(x / 10.0)) * 10

def calc_agg_dem(aggre_demand, roundup_flag = True):
    
    if roundup_flag:
        target = roundup(sum([item.get('quantity') for item in aggre_demand]))
    else:
        target = sum([item.get('quantity') for item in aggre_demand])
    
    print(f'Total Aggregated demand is {target} MT.')
    
    return target

def truck(data_truck, selectedDeliveryDate):
    
    truck_df = pd.DataFrame(data_truck)
    
    format='%d-%m-%Y'
    
    # DELIVERY START DATE
    from_date = selectedDeliveryDate.get('from')
    startdate = datetime(year=from_date.get('year'), month=from_date.get('month'), day=from_date.get('day'))
    
    # DELIVERY END DATE
    to_date = selectedDeliveryDate.get('to')
    enddate = datetime(year=to_date.get('year'), month=to_date.get('month'), day=to_date.get('day'))

    
    truck_df['availabilityDate'] = pd.to_datetime(truck_df['availabilityDate'], format=format)
    
    # SELECT TRUCKS WITHIN WEEK RANGE
    selected_trucks_df = truck_df.loc[(truck_df['availabilityDate'] >= startdate) & (truck_df['availabilityDate'] <= enddate)]
    
    # PUT TRUCK SIZES INTO LIST FOR PERMUTATIONS LATER
    truck_size = selected_trucks_df['truckSize'].to_list()
    print(f'Total available selected truck weight is: {sum(truck_size)} MT')
    # logging.debug('Add: {} + {} = {}'.format(num_1, num_2, add_result))
    
    rentalRate = selected_trucks_df['rentalRate'].to_list()
    
    return rentalRate, truck_size


@timer
def gekko2(target, rentalRate, truck_size_list):
    
    # cost, target, truck_size_list = get_data()
    
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
    x = np.array([m.Var(value=1,lb=0,ub=1,integer=True) for _ in range(len(truck_size_list))])
    # x = [m.Var(value=1,lb=0,ub=1,integer=True) for _ in range(len(truck_size_list))]
    
    # Equations
    # m.Equation(sum([i*j for i,j in zip(x,truck_size_list)])>=target)
    m.Equation(np.multiply(x,truck_size_list).sum()>target)     # Equation in numpy array
    
    # Objectives
    m.Obj(np.multiply(x,rentalRate).sum())      # Equation in numpy array
    # m.Obj(sum([i*j for i,j in zip(x,rentalRate)]))
    
    # Solve
    m.solve(disp=False) # Solve
    print('Results')
    
    selection = [i.value[0] for i in x]
    print('x: ' + str(selection))

    print('Objective: ' + str(m.options.objfcnval))
    
    total = sum([i*j for i,j in zip (selection,truck_size_list)])
    print(f'Total truck selected is: {total} MT')

def main():
    
    aggre_demand, selectedDeliveryDate, data_truck = get_data()
    
    target = calc_agg_dem(aggre_demand, roundup_flag = False)
    
    rentalRate, truck_size_list = truck(data_truck, selectedDeliveryDate)
    print(truck_size_list)
    print(rentalRate)
    
    gekko2(target, rentalRate, truck_size_list)
    
if __name__ == '__main__':
    
    main()