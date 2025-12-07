import math
from typing import List, Tuple, Optional
from fractions import Fraction

Numero = Fraction
Matriz = List[List[Numero]]

def _to_frac(v):
    try: return Fraction(str(v))
    except: return Fraction(0)

def copy_m(M): return [[_to_frac(x) for x in r] for r in M]
def ident(n): return [[Fraction(1) if i==j else Fraction(0) for j in range(n)] for i in range(n)]
def zeros(r, c): return [[Fraction(0) for _ in range(c)] for _ in range(r)]

def fmt_val(v):
    if v.denominator == 1: return str(v.numerator)
    return f"{v.numerator}/{v.denominator}"

def fmt_paso(M):
    if not M: return ""
    cols_w = [max(len(fmt_val(x)) for x in col) for col in zip(*M)]
    lines = []
    for row in M:
        lines.append("  [ " + "  ".join(f"{fmt_val(x):>{cols_w[i]}}" for i,x in enumerate(row)) + " ]")
    return "\n".join(lines)

# --- Operaciones Básicas ---
def sumar_matrices_dos(A, B):
    if len(A)!=len(B) or len(A[0])!=len(B[0]): raise ValueError("Dimensiones distintas")
    R = [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
    return R, ["Suma A + B:\n" + fmt_paso(R)]

def restar_matrices_dos(A, B):
    if len(A)!=len(B) or len(A[0])!=len(B[0]): raise ValueError("Dimensiones distintas")
    R = [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
    return R, ["Resta A - B:\n" + fmt_paso(R)]

def multiplicar_matrices(A, B):
    if len(A[0])!=len(B): raise ValueError("Incompatibles")
    C = zeros(len(A), len(B[0]))
    pasos = ["Inicio Multiplicación:\n" + fmt_paso(A) + "\n  X\n" + fmt_paso(B) + "\n"]
    for i in range(len(A)):
        for j in range(len(B[0])):
            s = 0
            for k in range(len(A[0])): s += A[i][k]*B[k][j]
            C[i][j] = s
    pasos.append("Matriz Resultante:\n" + fmt_paso(C))
    return C, pasos

def transpuesta(A):
    T = [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]
    return T, ["Transpuesta:\n" + fmt_paso(T)]

# --- GAUSS (REF) y GAUSS-JORDAN (RREF) ---

def ref(A):
    """Forma Escalonada (Row Echelon Form) - Solo ceros abajo."""
    M = copy_m(A)
    rows, cols = len(M), len(M[0])
    pasos = [f"Matriz Inicial:\n{fmt_paso(M)}\n"]
    r = 0
    for c in range(cols):
        if r >= rows: break
        pivot = r
        while pivot < rows and M[pivot][c] == 0: pivot += 1
        if pivot < rows:
            if pivot != r:
                M[r], M[pivot] = M[pivot], M[r]
                pasos.append(f"⬇ Intercambio F{r+1} <-> F{pivot+1}:\n{fmt_paso(M)}\n")
            
            # Normalizar (Opcional en Gauss puro, pero recomendado)
            piv_val = M[r][c]
            if piv_val != 1:
                for j in range(c, cols): M[r][j] /= piv_val
                pasos.append(f"➗ F{r+1} / {fmt_val(piv_val)} (Pivote=1):\n{fmt_paso(M)}\n")
            
            # Eliminar SOLO ABAJO
            cambio = False
            for i in range(r + 1, rows):
                if M[i][c] != 0:
                    f = M[i][c]
                    for j in range(c, cols): M[i][j] -= f * M[r][j]
                    pasos.append(f"➖ F{i+1} - ({fmt_val(f)})*F{r+1}")
                    cambio = True
            if cambio: pasos.append(f"   Estado (REF):\n{fmt_paso(M)}\n")
            r += 1
    pasos.append(f"✅ Forma Escalonada (REF) Final:\n{fmt_paso(M)}")
    return M, pasos

def rref(A):
    """Forma Escalonada Reducida (Reduced Row Echelon Form) - Ceros arriba y abajo."""
    M = copy_m(A)
    rows, cols = len(M), len(M[0])
    pasos = [f"Matriz Inicial:\n{fmt_paso(M)}\n"]
    r = 0
    for c in range(cols):
        if r >= rows: break
        pivot = r
        while pivot < rows and M[pivot][c] == 0: pivot += 1
        if pivot < rows:
            if pivot != r:
                M[r], M[pivot] = M[pivot], M[r]
                pasos.append(f"⬇ Intercambio F{r+1} <-> F{pivot+1}:\n{fmt_paso(M)}\n")
            piv_val = M[r][c]
            if piv_val != 1:
                for j in range(c, cols): M[r][j] /= piv_val
                pasos.append(f"➗ F{r+1} / {fmt_val(piv_val)} (Pivote=1):\n{fmt_paso(M)}\n")
            cambio = False
            for i in range(rows): # Eliminar ARRIBA Y ABAJO
                if i != r and M[i][c] != 0:
                    f = M[i][c]
                    for j in range(c, cols): M[i][j] -= f * M[r][j]
                    pasos.append(f"➖ F{i+1} - ({fmt_val(f)})*F{r+1}")
                    cambio = True
            if cambio: pasos.append(f"   Estado (RREF):\n{fmt_paso(M)}\n")
            r += 1
    pasos.append(f"✅ RREF Final:\n{fmt_paso(M)}")
    return M, pasos

rref_con_pasos = rref

# --- SOLUCIONADORES DE SISTEMAS ---

def resolver_gauss(A, b):
    # 1. Matriz Aumentada
    M = [row + [val_b] for row, val_b in zip(A, b)]
    pasos_totales = [f"Matriz Aumentada [A|b]:\n{fmt_paso(M)}\n"]
    
    # 2. Gauss (REF)
    M_ref, pasos_ref = ref(M)
    pasos_totales.extend(pasos_ref)
    
    # 3. Sustitución hacia atrás
    n = len(M_ref)
    # Check consistencia básica (fila de ceros = algo)
    for i in range(n):
        if all(M_ref[i][j]==0 for j in range(n)) and M_ref[i][-1]!=0:
            return None, pasos_totales + ["❌ Sistema Inconsistente (0 != k)"]

    x = [Fraction(0)] * n
    pasos_totales.append("\nSustitución Hacia Atrás:")
    
    for i in range(n-1, -1, -1):
        val = M_ref[i][-1]
        txt = f"x{i+1} = ({fmt_val(val)}"
        for j in range(i+1, n):
            val -= M_ref[i][j] * x[j]
            txt += f" - {fmt_val(M_ref[i][j])}*{fmt_val(x[j])}"
        
        if M_ref[i][i] == 0: continue # Variable libre o error
        x[i] = val / M_ref[i][i]
        pasos_totales.append(f"{txt}) / {fmt_val(M_ref[i][i])} = {fmt_val(x[i])}")
        
    return x, pasos_totales

def resolver_gauss_jordan(A, b):
    M = [row + [val_b] for row, val_b in zip(A, b)]
    pasos_totales = [f"Matriz Aumentada [A|b]:\n{fmt_paso(M)}\n"]
    
    # RREF directa
    M_rref, pasos_rref = rref(M)
    pasos_totales.extend(pasos_rref)
    
    n = len(M_rref)
    for i in range(n):
        if all(M_rref[i][j]==0 for j in range(n)) and M_rref[i][-1]!=0:
            return None, pasos_totales + ["❌ Sistema Inconsistente"]
            
    x = [row[-1] for row in M_rref]
    return x, pasos_totales

# --- Otras ---
def determinante(A):
    n = len(A)
    if n != len(A[0]): raise ValueError("No cuadrada")
    M = copy_m(A)
    det = Fraction(1)
    pasos = [f"Gauss para Determinante:\n{fmt_paso(M)}\n"]
    for i in range(n):
        p = i
        while p < n and M[p][i] == 0: p += 1
        if p == n: return Fraction(0), pasos + ["Columna 0 -> Det=0"]
        if p != i:
            M[i], M[p] = M[p], M[i]
            det *= -1
            pasos.append(f"Intercambio F{i+1}-F{p+1} (Det invierte signo)")
        piv = M[i][i]
        det *= piv
        for j in range(i+1, n):
            f = M[j][i] / piv
            for k in range(i, n): M[j][k] -= f * M[i][k]
    pasos.append(f"Multiplicación diagonal = {fmt_val(det)}")
    return det, pasos

def matriz_inversa(A):
    n = len(A)
    if n != len(A[0]): raise ValueError("No cuadrada")
    M = [r + row for r, row in zip(A, ident(n))]
    pasos = [f"Aumentada [A|I]:\n{fmt_paso(M)}\n"]
    # Reusamos lógica de rref para pasos limpios
    R, p = rref(M)
    pasos += p
    res = [row[n:] for row in R]
    return res, pasos

def regla_cramer(A, b):
    detA, pA = determinante(A)
    pasos = ["1. Det(A):\n" + "\n".join(pA) + f"\nResultado Det(A) = {fmt_val(detA)}\n"]
    if detA == 0: return None, pasos + ["Det 0, Cramer falla"]
    n = len(A)
    sol = []
    for i in range(n):
        Ai = copy_m(A)
        for j in range(n): Ai[j][i] = b[j]
        di, _ = determinante(Ai)
        sol.append(di/detA)
        pasos.append(f"x{i+1} = Det(A{i+1}) / Det(A) = {fmt_val(di)} / {fmt_val(detA)} = {fmt_val(sol[-1])}")
    return sol, pasos

def rango_matriz(A):
    R, _ = rref(A)
    r = sum(1 for row in R if any(x!=0 for x in row))
    return r, [f"RREF:\n{fmt_paso(R)}\nFilas no nulas = {r}"]