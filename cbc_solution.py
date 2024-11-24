# cbc_solution.py
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, NonNegativeIntegers, SolverFactory, RangeSet, NonNegativeReals, value
import numpy as np

def cbc_optimize(env):
    model = ConcreteModel()
    T = env.max_steps
    max_inventory = env.max_inventory
    max_order = env.max_order
    
    # Sets
    model.T = RangeSet(0, T - 1)
    
    # Parameters
    holding_cost = env.holding_cost
    shortage_cost = env.shortage_cost
    selling_price = env.selling_price
    ordering_cost = env.ordering_cost
    initial_inventory = env.max_inventory // 2
    demand = np.random.randint(0, 20, size=T)
    
    # Variables
    model.order = Var(model.T, domain=NonNegativeIntegers, bounds=(0, max_order))
    model.inventory = Var(model.T, domain=NonNegativeReals, bounds=(0, max_inventory))
    model.shortage = Var(model.T, domain=NonNegativeReals)
    model.sales = Var(model.T, domain=NonNegativeReals)
    
    # Objective: Maximize total profit
    def obj_expression(m):
        revenue = sum(selling_price * m.sales[t] for t in m.T)
        costs = sum(
            ordering_cost * m.order[t] +
            holding_cost * m.inventory[t] +
            shortage_cost * m.shortage[t]
            for t in m.T
        )
        return revenue - costs
    model.obj = Objective(rule=obj_expression, sense=-1)  # Minimize negative profit
    
    # Constraints
    def inventory_balance_rule(m, t):
        if t == 0:
            return m.inventory[t] == initial_inventory + m.order[t] - m.sales[t]
        else:
            return m.inventory[t] == m.inventory[t-1] + m.order[t] - m.sales[t]
    model.inventory_balance = Constraint(model.T, rule=inventory_balance_rule)
    
    def sales_rule(m, t):
        return m.sales[t] + m.shortage[t] == demand[t]
    model.sales_constraint = Constraint(model.T, rule=sales_rule)
    
    # Solve the model
    solver = SolverFactory('cbc')
    result = solver.solve(model)
    
    # Extract the optimal actions
    optimal_orders = [int(value(model.order[t])) for t in model.T]
    
    return optimal_orders, demand
