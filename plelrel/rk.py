import numpy

def euler(simulation, cons, prim, aux):
    dt = simulation.dt
    rhs = simulation.rhs
    return cons + dt * rhs(cons, prim, aux, simulation)

def rk2(simulation, cons, prim, aux):
    dt = simulation.dt
    rhs = simulation.rhs
    cons1 = cons + dt * rhs(cons, prim, aux, simulation)
    cons1 = simulation.bcs(cons1, simulation.grid.Npoints, simulation.grid.Ngz)
    prim1, aux1 = simulation.model.cons2all(cons1, prim)
    return 0.5 * (cons + cons1 + dt * rhs(cons1, prim1, aux1, simulation))

def rk3(simulation, cons, prim, aux):
    dt = simulation.dt
    rhs = simulation.rhs
    cons1 = cons + dt * rhs(cons, prim, aux, simulation)
    cons1 = simulation.bcs(cons1, simulation.grid.Npoints, simulation.grid.Ngz)
    if simulation.fix_cons:
        cons1 = simulation.model.fix_cons(cons1)
    prim1, aux1 = simulation.model.cons2all(cons1, prim)
    cons2 = (3 * cons + cons1 + dt * rhs(cons1, prim1, aux1, simulation)) / 4
    cons2 = simulation.bcs(cons2, simulation.grid.Npoints, simulation.grid.Ngz)
    if simulation.fix_cons:
        cons2 = simulation.model.fix_cons(cons2)
    prim2, aux2 = simulation.model.cons2all(cons2, prim1)
    return (cons + 2 * cons2 + 2 * dt * rhs(cons2, prim2, aux2, simulation)) / 3
