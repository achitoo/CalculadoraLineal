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
    """Formatea fracción a string bonito (5/3)."""
    if v.denominator == 1: return str(v.numerator)
    return f"{v.numerator}/{v.denominator}"

def fmt_paso(M):
    """Crea una representación visual de texto de la matriz."""
    if not M: return ""
    cols_w = [max(len(fmt_val(x)) for x in col) for col in zip(*M)]
    lines = []
    for row in M:
        lines.append("  [ " + "  ".join(f"{fmt_val(x):>{cols_w[i]}}" for i,x in enumerate(row)) + " ]")
    return "\n".join(lines)

# --- Operaciones ---

def sumar_matrices_dos(A, B):
    if len(A)!=len(B) or len(A[0])!=len(B[0]): raise ValueError("Dimensiones distintas")
    R = [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
    return R, ["Suma directa A + B:\n" + fmt_paso(R)]

def restar_matrices_dos(A, B):
    if len(A)!=len(B) or len(A[0])!=len(B[0]): raise ValueError("Dimensiones distintas")
    R = [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
    return R, ["Resta directa A - B:\n" + fmt_paso(R)]

def multiplicar_matrices(A, B):
    if len(A[0])!=len(B): raise ValueError("Incompatibles")
    C = zeros(len(A), len(B[0]))
    pasos = ["Inicio Multiplicación:\n" + fmt_paso(A) + "\n  X\n" + fmt_paso(B) + "\n"]
    for i in range(len(A)):
        for j in range(len(B[0])):
            term = []
            s = 0
            for k in range(len(A[0])):
                val = A[i][k]*B[k][j]
                s += val
                term.append(f"({fmt_val(A[i][k])}·{fmt_val(B[k][j])})")
            C[i][j] = s
            if len(A)*len(B[0]) < 12: 
                pasos.append(f"C[{i+1},{j+1}] = {' + '.join(term)} = {fmt_val(s)}")
    
    pasos.append("\nMatriz Resultante:\n" + fmt_paso(C))
    return C, pasos

def transpuesta(A):
    T = [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]
    return T, ["Transpuesta (Filas -> Cols):\n" + fmt_paso(T)]

def rref(A):
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
                pasos.append(f"⬇ Intercambio Fila {r+1} <-> Fila {pivot+1}:\n{fmt_paso(M)}\n")
            
            piv_val = M[r][c]
            if piv_val != 1:
                for j in range(c, cols): M[r][j] /= piv_val
                pasos.append(f"➗ Fila {r+1} dividida por {fmt_val(piv_val)} (Pivote=1):\n{fmt_paso(M)}\n")
            
            cambio = False
            for i in range(rows):
                if i != r and M[i][c] != 0:
                    f = M[i][c]
                    for j in range(c, cols): M[i][j] -= f * M[r][j]
                    pasos.append(f"➖ Fila {i+1} - ({fmt_val(f)}) * Fila {r+1}")
                    cambio = True
            
            if cambio:
                pasos.append(f"   Estado tras limpiar columna {c+1}:\n{fmt_paso(M)}\n")
            r += 1
            
    pasos.append(f"✅ Forma Escalonada Reducida (RREF) Final:\n{fmt_paso(M)}")
    return M, pasos

rref_con_pasos = rref 

def determinante(A):
    n = len(A)
    if n != len(A[0]): raise ValueError("No cuadrada")
    M = copy_m(A)
    det = Fraction(1)
    pasos = [f"Matriz Inicial (Método Gauss):\n{fmt_paso(M)}\n"]
    
    for i in range(n):
        p = i
        while p < n and M[p][i] == 0: p += 1
        
        if p == n: 
            pasos.append("❌ Columna de ceros encontrada -> Determinante = 0")
            return Fraction(0), pasos
            
        if p != i:
            M[i], M[p] = M[p], M[i]
            det *= -1
            pasos.append(f"⬇ Intercambio F{i+1} <-> F{p+1} (Det cambia signo):\n{fmt_paso(M)}\n")
            
        piv = M[i][i]
        det *= piv
        pasos.append(f"ℹ Pivote en ({i+1},{i+1}) es {fmt_val(piv)}. Det acumulado = {fmt_val(det)}")
        
        for j in range(i+1, n):
            if M[j][i] != 0:
                f = M[j][i] / M[i][i]
                for k in range(i, n): M[j][k] -= f * M[i][k]
                
    return det, pasos

def matriz_inversa(A):
    n = len(A)
    if n != len(A[0]): raise ValueError("No cuadrada")
    M = [r + row for r, row in zip(A, ident(n))]
    pasos = [f"Matriz Aumentada [A|I]:\n{fmt_paso(M)}\n"]
    
    rows, cols = n, 2*n
    r = 0
    for c in range(n):
        if r >= rows: break
        pivot = r
        while pivot < rows and M[pivot][c] == 0: pivot += 1
        if pivot == rows: return None, pasos + ["❌ Matriz Singular (Pivote 0)"]
        
        if pivot != r:
            M[r], M[pivot] = M[pivot], M[r]
            pasos.append(f"⬇ Intercambio F{r+1} <-> F{pivot+1}:\n{fmt_paso(M)}\n")
            
        piv = M[r][c]
        inv = 1/piv
        for j in range(cols): M[r][j] *= inv
        pasos.append(f"➗ Fila {r+1} / {fmt_val(piv)}:\n{fmt_paso(M)}\n")
        
        for k in range(rows):
            if k != r:
                f = M[k][c]
                for j in range(cols): M[k][j] -= f * M[r][j]
                
        pasos.append(f"➖ Eliminación en columna {c+1}:\n{fmt_paso(M)}\n")
        r += 1
        
    res = [row[n:] for row in M]
    pasos.append(f"✅ Inversa Final:\n{fmt_paso(res)}")
    return res, pasos

def regla_cramer(A, b):
    detA, _ = determinante(A)
    pasos = [f"1. Determinante Principal Det(A) = {fmt_val(detA)}"]
    if detA == 0: return None, pasos + ["❌ Det(A) es 0. Cramer no aplica."]
    
    n = len(A)
    sol = []
    for i in range(n):
        Ai = copy_m(A)
        for j in range(n): Ai[j][i] = b[j]
        di, _ = determinante(Ai)
        res = di/detA
        pasos.append(f"\n2.{i+1} Calculando x{i+1}:")
        pasos.append(f"   Matriz A{i+1} (Reemplazando col {i+1} por b):\n{fmt_paso(Ai)}")
        pasos.append(f"   Det(A{i+1}) = {fmt_val(di)}")
        pasos.append(f"   x{i+1} = {fmt_val(di)} / {fmt_val(detA)} = {fmt_val(res)}")
        sol.append(res)
    
    return sol, pasos

def rango_matriz(A):
    R, _ = rref(A)
    r = sum(1 for row in R if any(x!=0 for x in row))
    return r, [f"Rango calculado contando filas no nulas en RREF: {r}"]