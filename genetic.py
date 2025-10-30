import numpy as np

# ----------------- Modelo ideal do nozzle ----------------- #
def area_mach(M, gamma):
    term = (2/(gamma+1))*(1+(gamma-1)*M**2/2)
    return (1/M) * term**((gamma+1)/(2*(gamma-1)))

def solve_Me_from_eps(eps, gamma, M_lo=1.01, M_hi=20.0, itmax=100):
    # bissecção na raiz de A/A* - eps = 0 (ramo supersónico)
    f = lambda M: area_mach(M, gamma) - eps
    a, b = M_lo, M_hi
    fa, fb = f(a), f(b)
    if fa*fb > 0:
        return np.nan  # falha
    for _ in range(itmax):
        m = 0.5*(a+b)
        fm = f(m)
        if abs(fm) < 1e-8: return m
        if fa*fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5*(a+b)

def thrust_coefficient(eps, gamma, pc, pa):
    Me = solve_Me_from_eps(eps, gamma)
    if not np.isfinite(Me): return np.nan, np.nan, np.nan
    pe_over_pc = (1 + (gamma-1)*Me**2/2)**(-gamma/(gamma-1))
    pe = pe_over_pc * pc
    term1 = np.sqrt( (2*gamma**2/(gamma-1)) * (2/(gamma+1))**((gamma+1)/(gamma-1)) * (1 - pe_over_pc**((gamma-1)/gamma)) )
    Cf = term1 + (pe - pa)/pc * eps
    return Cf, pe, Me

def thrust(Cf, pc, At):
    return Cf * pc * At  # modelo ideal

# ----------------- GA simples (ε, A_t) ----------------- #
def ga_nozzle(
    F_target=3000.0, Itot=5000.0,
    pc=4.0e6, pa=1.013e5, gamma=1.22,
    eps_bounds=(2.0, 80.0), At_bounds=(2e-5, 2e-3),
    pop_size=120, n_gen=200, cx_rate=0.9, mut_rate=0.2, elitism=2, seed=7
):
    rng = np.random.default_rng(seed)

    def sample():
        eps = rng.uniform(*eps_bounds)
        At  = rng.uniform(*At_bounds)
        return np.array([eps, At])

    def mutate(ind):
        if rng.random() < mut_rate:
            # mutação multiplicativa limitada
            ind[0] *= np.exp(rng.normal(0, 0.15))
            ind[1] *= np.exp(rng.normal(0, 0.15))
        # clamping
        ind[0] = np.clip(ind[0], *eps_bounds)
        ind[1] = np.clip(ind[1], *At_bounds)
        return ind

    def crossover(a, b):
        alpha = rng.uniform(-0.2, 1.2, size=2)
        c1 = alpha*a + (1-alpha)*b
        c2 = alpha*b + (1-alpha)*a
        # clamp
        c1[0] = np.clip(c1[0], *eps_bounds); c1[1] = np.clip(c1[1], *At_bounds)
        c2[0] = np.clip(c2[0], *eps_bounds); c2[1] = np.clip(c2[1], *At_bounds)
        return c1, c2

    def fitness(ind):
        eps, At = ind
        Cf, pe, Me = thrust_coefficient(eps, gamma, pc, pa)
        if not np.isfinite(Cf):  # penalizar falhas
            return -1e9
        F = thrust(Cf, pc, At)
        tb = Itot / max(F, 1e-6)
        # perdas normalizadas
        eF  = abs(F - F_target)/F_target
        ePe = abs(pe - pa)/pa
        # penalizar tempos de queima muito longes do alvo implícito (Itot/F_target)
        tb_target = Itot / F_target
        et = abs(tb - tb_target)/tb_target
        # função objetivo: minimizar combinação
        J = 1.0*eF + 0.3*ePe + 0.2*et
        return -J  # GA maximiza fitness

    # inicialização
    pop = np.array([sample() for _ in range(pop_size)])
    fit = np.array([fitness(ind) for ind in pop])

    for _ in range(n_gen):
        # elitismo
        elite_idx = np.argsort(-fit)[:elitism]
        elites = pop[elite_idx].copy()
        # seleção por torneio
        new_pop = []
        while len(new_pop) < pop_size - elitism:
            k = 3
            cand = rng.integers(0, pop_size, size=(2, k))
            p1 = pop[cand[0][np.argmax(fit[cand[0]])]]
            p2 = pop[cand[1][np.argmax(fit[cand[1]])]]
            # crossover + mutação
            if rng.random() < cx_rate:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            c1 = mutate(c1); c2 = mutate(c2)
            new_pop.extend([c1, c2])
        pop = np.vstack([elites, np.array(new_pop[:pop_size-elitism])])
        fit = np.array([fitness(ind) for ind in pop])

    # melhor
    i = int(np.argmax(fit))
    eps, At = pop[i]
    Cf, pe, Me = thrust_coefficient(eps, gamma, pc, pa)
    F = thrust(Cf, pc, At)
    tb = Itot / F
    return {
        "epsilon": eps,
        "At [m^2]": At,
        "Cf": Cf,
        "F [N]": F,
        "pe [Pa]": pe,
        "Me": Me,
        "tb [s]": tb,
        "pc [Pa]": pc,
        "pa [Pa]": pa,
        "gamma": gamma
    }

if __name__ == "__main__":
    res = ga_nozzle(
        F_target=3000.0, Itot=5000.0,
        pc=3.0e6, pa=1.013e5, gamma=1.22,  # ajusta conforme o teu caso
        eps_bounds=(3.0, 60.0), At_bounds=(5e-5, 1e-3),
        pop_size=150, n_gen=300
    )
    for k, v in res.items():
        print(f"{k}: {v}")
