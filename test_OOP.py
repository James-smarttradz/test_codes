# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 22:49:11 2021

@author: James Ang
"""

class Employee:
    
    num_emp = 0
    raise_amt = 1.04
    
    def __init__(self, first_name, last_name, salary):
        
        self.first_name = first_name
        self.last_name = last_name
        self.salary = salary
        Employee.num_emp +=1
        
    def __call__(self):
        print('This is call method')
    
    
    # These are regular methods
    
    def fullname(self):
        
        return self.first_name + ' ' + self.last_name
    
    def apply_raise(self):
        self.salary = self.salary * self.raise_amt

    
    
    # here we're working with the class instead of the instance
    @classmethod
    def set_raise_amt(cls,amount):
        cls.raise_amt = amount
        
    # Using class methods as alternative constructors
    @classmethod
    def from_string(cls, emp_str):
        first, last, pay = emp_str.split('-')
        pay = int(pay)
        return cls(first,last,pay)
    
    # Below is static class
    @staticmethod
    def is_workday(day):
        if day.weekday == 5 or day.weekday ==6:
            return False
        return True
    
    def __repr__(self):
        return "Employee ('{}','{}',{})".format(self.first_name, self.last_name, self.salary)

    def __str__(self):
        return "{} - {}".format(self.fullname(), self.salary)
    
emp1 = Employee('Stan', 'Lee', 50000)
emp2 = Employee('Roberto', 'Lee', 40000)

# print(emp1.first_name)
# print(emp1.fullname())
# print('raise: ' + str(emp1.apply_raise()))

# print(emp2.__dict__)
# print(Employee.__dict__)

# print(Employee.num_emp)


# Employee.set_raise_amt(1.06)
# print(Employee.raise_amt)
# print(emp1.raise_amt)
# print(emp2.raise_amt)

# =============================================================================
# To test class methods
# =============================================================================
emp_str_1 = 'Bob-Love-100000'
emp3 = Employee.from_string(emp_str_1)

print(emp3.salary)
emp3.apply_raise()
print(emp3.salary)


# =============================================================================
# To test static methods
# =============================================================================

import datetime

my_date = datetime.date(2020, 2, 5)
Employee.is_workday(my_date)


# =============================================================================
# To show inheritance - Creating subclasses
# Inheritance will not break anything in the parent class
# =============================================================================

class Developer(Employee): # Just by adding Employee, we get all the inherited class for free
    
    # if want to change for developers raise to 10%
    raise_amt = 1.10
    
    def __init__(self, first_name, last_name, salary, prog_lang):
        
        super().__init__(first_name, last_name, salary)
        self.prog_lang = prog_lang

# Code below specific for Manager

class Manager(Employee): # Just by adding Employee, we get all the inherited class for free
    
    
    def __init__(self, first_name, last_name, salary, employee=None):
        
        super().__init__(first_name, last_name, salary)
        
        if employee == None:
            self.employee = []
        else:
            self.employee = employee
        
    def add_emp(self, emp):
        
        if emp not in self.employee:
            self.employee.append(emp)
            
    def rem_emp(self,emp):
        
        if emp in self.employee:
            self.employee.remove(emp)
            
    def print_emp(self):
        
        for emp in self.employee:
            print('--> ',emp.fullname())
    


dev4 = Developer('Janice', 'Sze', 80000, 'Python')

# print(help(Developer))

print(dev4.salary)
dev4.apply_raise()
print(dev4.salary)
print(dev4.prog_lang)

mgr5 = Manager('Sophie', 'Chung', 2e6, [emp1])
mgr5.print_emp()
mgr5.add_emp(dev4)
mgr5.print_emp()
mgr5.rem_emp(emp1)
mgr5.print_emp()

print(isinstance(mgr5, Manager))

# =============================================================================
# Special Methods
# =============================================================================

# __repr__

print(emp1)
print(emp2)

# __str__
print(str(emp1))
print(repr(emp1))

print(emp1.__str__())
print(emp1.__repr__())