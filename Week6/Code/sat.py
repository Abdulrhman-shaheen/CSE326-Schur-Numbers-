from z3 import *
import time
import sys

# Schur numbers
# 1, 4, 13, 44, 160

# 4,   44, sat-encoding => 0.47s
# 4,   45, sat-encoding => 585.15s
# 4, 1000, sat-encoding => 783.58s
# 4,   45, sat-encoding => 58.12s (cvc5)
# 4,   45, sat-encoding => 82.79s (custom dimacs glucose)
# 4,   45, sat-encoding => 36.12s (custom dimacs glucose-syrup)


# 5, 160 


k = 4
n = 44

model = "sat"

if len(sys.argv) > 2:
    n = int(sys.argv[1])
    k = int(sys.argv[2])
if len(sys.argv) > 3:
    model = sys.argv[3]




def z3_to_cnf_clauses(formula):
    """Convert a Z3 formula into a list of CNF clauses."""
    # Tactic to convert the formula to CNF
    goal = Goal()
    goal.add(formula)
    cnf = Then(Tactic('tseitin-cnf'), Tactic('simplify')).apply(goal)
    assert len(cnf) == 1, "Expected a single goal"
    clauses = []
    for clause in cnf[0]:
        # Collect literals for each clause
        literals = []
        if is_or(clause):
            for literal in clause.children():
                literals.append(literal)
        else:
            literals.append(clause)
        clauses.append(literals)
    return clauses

def write_dimacs(clauses, var_mapping, filename="formula.dimacs"):
    """Write clauses to a DIMACS file."""
    with open(filename, 'w') as f:
        # DIMACS header
        num_vars = len(var_mapping)
        num_clauses = len(clauses)
        f.write(f"p cnf {num_vars} {num_clauses}\n")
        
        # Write each clause in DIMACS format
        for clause in clauses:
            clause_str = ""
            for literal in clause:
                var = literal.decl().name()
                if var == 'not':
                    var = literal.children()[0].decl().name()
                    var_id = -var_mapping[var]
                else:
                    var_id = var_mapping[var]
                clause_str += f"{var_id} "
            clause_str += "0\n"
            f.write(clause_str)






s = Solver()

if model == "sat":
    colors = [
        [Bool('color_%d_%d' % (i, j)) for j in range(k)]
        for i in range(n)
    ]

    # exactly one color per number
    for i in range(n):
        # at least one color
        s.add(Or(colors[i]))
        # at most one color
        s.add(Not(And(colors[i])))
        
    for a in range(1,n+1):
        for b in range(a,n+1):
            c = a + b
            if c <= n:
                ai = a-1
                bi = b-1
                ci = c-1
                # not all three colors are the same
                for i in range(k):
                    s.add(Not(And(colors[ai][i], colors[bi][i], colors[ci][i])))
                    
    # heuristics oBdA
    s.add(colors[1-1][0]) # 1 -> 0
    s.add(colors[2-1][1]) # 2 -> 1

    # no two consecutive numbers have color 0 
    # (otherwise 1 + i = i+1)
    for i in range(n-1):
        s.add(Not(And(colors[i][0], colors[i+1][0])))
    # same for color 1 with 2
    # 2 + i = i+2
    for i in range(n-2):
        s.add(Not(And(colors[i][1], colors[i+2][1])))
elif model == "smt":
    colors = [Int('color_%d' % i) for i in range(n)]

    for i in range(n):
        s.add(And(colors[i] >= 0, colors[i] < k))
        
    for a in range(1,n+1):
        for b in range(a,n+1):
            c = a + b
            if c <= n:
                # print(f"{a} + {b} = {c}")
                # not all three colors are the same
                s.add(Or(
                    colors[a-1] != colors[b-1], 
                    colors[a-1] != colors[c-1]
                ))
                # s.add(Not(And(
                #     colors[a-1] == colors[b-1],
                #     colors[a-1] == colors[c-1]
                # )))
else:
    raise ValueError("Invalid model")
                
                
print(f"Writing files for n={n}, k={k}")

# export to smt2
with open("schur_%d_%d_%s.smt2" % (n, k, model), "w") as f:
    f.write(s.to_smt2())
# export to dimacs
# t = Then('simplify', 'bit-blast', 'tseitin-cnf')
# subgoal = t(s.assertions()[0])
# with open("schur_%d_%d.dimacs" % (n, k), "w") as f:
#     f.write(s.dimacs())
# with open("schur_%d_%d.dimacs" % (n, k), "w") as f:
#     f.write(subgoal.to_dimacs())

if model == "sat":
    # Get CNF clauses from the formula
    clauses = z3_to_cnf_clauses(And(s.assertions()))
    # Create a mapping from Z3 variables to integers
    var_mapping = {str(v): i+1 for i, v in enumerate(set(literal.decl() for clause in clauses for literal in clause))}
    # Write the DIMACS file
    write_dimacs(clauses, var_mapping, "schur_%d_%d.dimacs" % (n, k))
    # save var mapping to file
    with open("schur_%d_%d.var_mapping" % (n, k), "w") as f:
        for var, i in var_mapping.items():
            f.write(f"{var} {i}\n")
elif model == "smt":
    pass
else:
    raise ValueError("Invalid model")

print(f"Solving for n={n}, k={k} using {model}")
    
# dimacs = s.dimacs()
# read dimacs in again
# s = Solver()
# s.from_file("schur_%d_%d.dimacs" % (n, k))
            
t0 = time.time()
res = s.check()
t1 = time.time()

if res == sat:
    m = s.model()
    if model == "sat":
        colors = [
            (i+1,
                # find the i which is true
                next(j for j in range(k) if is_true(m.evaluate(colors[i][j])))
            ) for i in range(n)]
    elif model == "smt":
        print([m.evaluate(colors[i]) for i in range(n)])
    else:
        raise ValueError("Invalid model")
    print(f"Solution for n={n}, k={k} found in {t1-t0:.2f}s")
    # print(" ".join([str(colors[i]) for i in range(n)]))
    n_len = len(str(n))
    pad = lambda x, n: " "*(n-len(x)) + x
    print(" ".join([pad(str(i),n_len) for i, c in colors]))
    # color padded to number length
    print(" ".join([pad(str(c), n_len) for i, c in colors]))
else:
    # print("No solution for n=%d, k=%d" % (n, k))
    print(f"No solution for n={n}, k={k} in {t1-t0:.2f}s")