"""
Pequeño módulo plug&play para tu app Tkinter.
Proporciona:
  • AlgebraicKeypad: teclado algebraico para escribir expresiones tipo "A+B", "A-B", "A*B", "T(A)", "inv(A)".
  • EquationFillDialog: diálogo que calcula una expresión y la vuelca sobre una MatrixInput.
  • LinearSystemDialog: convierte un sistema lineal (texto) en matrices A y b.
No requiere dependencias externas. Usa Fraction del stdlib.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from fractions import Fraction
import ast
import re

# ------------------ utilidades -------------------------------------------------------------------

def is_scalar(x):
    return isinstance(x, (int, float, Fraction))

def mat_shape(M):
    return (len(M), len(M[0])) if M and isinstance(M[0], list) and M[0] else (0,0)

def mat_zeros(r, c):
    return [[Fraction(0) for _ in range(c)] for _ in range(r)]

def to_frac(x):
    if isinstance(x, Fraction): return x
    try: return Fraction(x)
    except: return Fraction(str(x))

def mat_copy(M):
    return [[to_frac(v) for v in row] for row in M]

def mat_add(A,B):
    if mat_shape(A) != mat_shape(B):
        raise ValueError("Dimensiones incompatibles para A+B")
    r,c = mat_shape(A)
    return [[A[i][j]+B[i][j] for j in range(c)] for i in range(r)]

def mat_sub(A,B):
    if mat_shape(A) != mat_shape(B):
        raise ValueError("Dimensiones incompatibles para A-B")
    r,c = mat_shape(A)
    return [[A[i][j]-B[i][j] for j in range(c)] for i in range(r)]

def mat_mul(A,B):
    # permite escalar*matriz, matriz*escalar y producto matricial
    if is_scalar(A) and not is_scalar(B):
        r,c = mat_shape(B)
        return [[to_frac(A)*B[i][j] for j in range(c)] for i in range(r)]
    if is_scalar(B) and not is_scalar(A):
        r,c = mat_shape(A)
        return [[A[i][j]*to_frac(B) for j in range(c)] for i in range(r)]
    # producto matricial
    ra, ca = mat_shape(A)
    rb, cb = mat_shape(B)
    if ca != rb: raise ValueError("Dimensiones incompatibles para A*B")
    C = mat_zeros(ra, cb)
    for i in range(ra):
        for j in range(cb):
            s = Fraction(0)
            for k in range(ca):
                s += A[i][k]*B[k][j]
            C[i][j] = s
    return C

def mat_T(A):
    r,c = mat_shape(A)
    return [[A[j][i] for j in range(r)] for i in range(c)]

def mat_eye(n):
    from fractions import Fraction
    return [[Fraction(1) if i==j else Fraction(0) for j in range(n)] for i in range(n)]

def mat_inv(A):
    # Gauss-Jordan para inversa (simple y suficiente para el diálogo)
    n,m = mat_shape(A)
    if n != m: raise ValueError("La matriz no es cuadrada para inv(A)")
    M = [row + eye for row, eye in zip(mat_copy(A), mat_eye(n))]
    rows, cols = n, 2*n
    r = 0
    for c in range(n):
        if r >= rows: break
        # buscar pivote
        pivot = r
        while pivot < rows and M[pivot][c] == 0:
            pivot += 1
        if pivot == rows:
            raise ValueError("Matriz singular (no invertible)")
        if pivot != r:
            M[r], M[pivot] = M[pivot], M[r]
        piv = M[r][c]
        inv = Fraction(1,1)/piv
        for j in range(cols): M[r][j] *= inv
        for i in range(rows):
            if i != r:
                f = M[i][c]
                if f != 0:
                    for j in range(cols): M[i][j] -= f*M[r][j]
        r += 1
    return [row[n:] for row in M]

# ------------------ evaluador seguro de expresiones ----------------------------------------------

_ALLOWED_BINOP = {ast.Add: mat_add, ast.Sub: mat_sub, ast.Mult: mat_mul}
_ALLOWED_UNARY = {ast.USub: lambda x: mat_mul(-1, x), ast.UAdd: lambda x: x}

def _eval(node, env):
    if isinstance(node, ast.Expression):
        return _eval(node.body, env)
    if isinstance(node, ast.BinOp):
        L = _eval(node.left, env)
        R = _eval(node.right, env)
        for k,fn in _ALLOWED_BINOP.items():
            if isinstance(node.op, k):
                return fn(L,R)
        raise ValueError("Operador no permitido")
    if isinstance(node, ast.UnaryOp):
        val = _eval(node.operand, env)
        for k,fn in _ALLOWED_UNARY.items():
            if isinstance(node.op, k): return fn(val)
        raise ValueError("Operador unario no permitido")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Función no permitida")
        fname = node.func.id
        if fname == 'T':
            if len(node.args)!=1: raise ValueError("T() espera 1 argumento")
            return mat_T(_eval(node.args[0], env))
        if fname == 'inv':
            if len(node.args)!=1: raise ValueError("inv() espera 1 argumento")
            return mat_inv(_eval(node.args[0], env))
        raise ValueError("Función no permitida: "+fname)
    if isinstance(node, ast.Name):
        if node.id in env: return env[node.id]
        raise ValueError(f"Símbolo desconocido: {node.id}")
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v,(int,float)): return Fraction(v)
        raise ValueError("Constante no válida")
    if isinstance(node, ast.Tuple):
        # permite vectores ((a,b),(c,d)) -> matriz 2x2
        rows = [ _eval(elt, env) for elt in node.elts ]
        # cada elemento puede ser un número o a su vez una tupla
        if all(isinstance(r, Fraction) for r in rows):
            # vector columna -> nx1
            return [[r] for r in rows]
        elif all(isinstance(r, tuple) for r in rows):
            # convertir tupla anidada a matriz
            M = []
            for t in rows:
                M.append(list(t))
            return M
        else:
            # si cada r es Fraction o lista de Fractions permitimos construir matriz
            M = []
            for r in rows:
                if isinstance(r, list): M.append(r)
                elif isinstance(r, Fraction): M.append([r])
                else: raise ValueError("Estructura de tuplas no válida")
            return M
    if isinstance(node, ast.List):
        # [[1,2],[3,4]]
        out = []
        for row in node.elts:
            rv = _eval(row, env)
            if isinstance(rv, list): out.append(rv)
            elif isinstance(rv, Fraction): out.append([rv])
            else: raise ValueError("Lista mal formada")
        return out
    if isinstance(node, ast.Dict):
        raise ValueError("Dict no permitido")
    if isinstance(node, ast.Subscript):
        raise ValueError("Subscripts no permitidos")
    raise ValueError("Expresión no soportada")

def eval_matrix_expr(expr, matrices_env):
    """
    Evalúa una expresión como "A+B", "2*A - T(B)", "inv(A)*B".
    matrices_env: diccionario con posibles llaves 'A' y 'B' (matrices tipo List[List[Fraction]]).
    """
    # normalizaciones comunes
    expr = expr.replace('×','*').replace('·','*')
    node = ast.parse(expr, mode='eval')
    return _eval(node, matrices_env)

# ------------------ parser de sistemas lineales ---------------------------------------------------

TOKEN = re.compile(r'([+-]?\d*\.?\d*)([a-zA-Z]+)')

def parse_linear_system(text):
    """
    Convierte un sistema escrito como:
        x + 3y = 7
        5x - y = 3
    en (A, b, variables)
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    left_terms = []
    right_vals = []
    vars_order = []
    # detectar todas las variables
    vars_set = set()
    for ln in lines:
        if '=' not in ln: raise ValueError("Cada línea debe contener '='")
        L, R = ln.split('=', 1)
        for coef, var in TOKEN.findall(L.replace(' ', '')):
            vars_set.add(var)
    vars_order = sorted(vars_set)  # abc...

    for ln in lines:
        L, R = [s.strip() for s in ln.split('=', 1)]
        coef_map = {v: Fraction(0) for v in vars_order}
        # normalizar signos como "+x" o "-y" -> "1x"/"-1y"
        L2 = L.replace(' ', '')
        # inserta + delante si el primer término es variable sin signo
        if L2 and L2[0] not in '+-':
            L2 = '+' + L2
        # capturar términos
        for m in TOKEN.finditer(L2):
            coef_raw, var = m.groups()
            if coef_raw in ('', '+', '-'):
                coef_raw = coef_raw + '1' if coef_raw != '' else '1'
            coef = Fraction(coef_raw)
            coef_map[var] += coef
        left_terms.append([coef_map[v] for v in vars_order])
        right_vals.append(Fraction(R))
    return left_terms, right_vals, vars_order

# ------------------ UI: diálogos ------------------------------------------------------------------

class AlgebraicKeypad(tk.Frame):
    """
    Teclado minimalista para construir expresiones matriciales.
    Provee callbacks 'on_accept' y 'on_cancel'.
    """
    def __init__(self, parent, on_accept, on_cancel=None, initial=''):
        super().__init__(parent, bg='white')
        self.on_accept = on_accept
        self.on_cancel = on_cancel
        self.entry = tk.Entry(self, font=('Consolas', 12))
        self.entry.insert(0, initial)
        self.entry.grid(row=0, column=0, columnspan=6, sticky='ew', pady=6, padx=6)
        self.columnconfigure(0, weight=1)
        buttons = [
            ('A','A'), ('B','B'), ('+','+'), ('-','-'), ('×','*'), ('·','*'),
            ('T( )','T('), ('inv( )','inv('), ('(', '('), (')', ')'),
            ('[','['), (']',']'), (',',','), ('1','1'), ('2','2'), ('3','3'),
            ('4','4'), ('5','5'), ('6','6'), ('7','7'), ('8','8'), ('9','9'), ('0','0'),
            ('DEL','<'), ('LIMPIAR','CLR'), ('OK','OK')
        ]
        r,c = 1,0
        for txt,val in buttons:
            b = tk.Button(self, text=txt, width=8, command=lambda v=val: self._press(v))
            b.grid(row=r, column=c, padx=2, pady=2)
            c += 1
            if c>=6: r += 1; c = 0

    def _press(self, v):
        if v == 'OK':
            if callable(self.on_accept): self.on_accept(self.entry.get())
            return
        if v == '<':
            cur = self.entry.get()
            self.entry.delete(0, 'end')
            self.entry.insert(0, cur[:-1])
            return
        if v == 'CLR':
            self.entry.delete(0,'end'); return
        # inserción normal
        self.entry.insert('end', v if v not in ('T(','inv(') else v)

class EquationFillDialog(tk.Toplevel):
    """
    Abre un diálogo con teclado algebraico para evaluar una expresión con A/B
    y volcar el resultado en un target (callable que recibe la matriz calculada).
    """
    def __init__(self, parent, getA, getB, on_fill, title="Rellenar con ecuación"):
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.getA = getA; self.getB = getB; self.on_fill = on_fill
        tk.Label(self, text="Escribe una expresión (usa A, B, +, -, *, T(), inv())").pack(padx=8, pady=4)
        self.err = tk.Label(self, text="", fg="red")
        self.err.pack()
        self.pad = AlgebraicKeypad(self, on_accept=self._accept)
        self.pad.pack(padx=6, pady=6)
        self.bind('<Return>', lambda e: self._accept(self.pad.entry.get()))

    def _accept(self, expr):
        try:
            env = {}
            try: env['A'] = self.getA()
            except: pass
            try: env['B'] = self.getB()
            except: pass
            M = eval_matrix_expr(expr, env)
            if not (isinstance(M, list) and M and isinstance(M[0], list)):
                raise ValueError("La expresión no produjo una matriz")
            self.on_fill(M)
            self.destroy()
        except Exception as e:
            self.err.config(text=str(e))

class LinearSystemDialog(tk.Toplevel):
    """
    Convierte un sistema en matrices y permite rellenar widgets MatrixInput A y b.
    """
    def __init__(self, parent, on_fillA, on_fillb, title="Sistema → Matrices (A,b)"):
        super().__init__(parent)
        self.title(title)
        self.resizable(True, True)
        tk.Label(self, text="Escribe un sistema, una ecuación por línea. Ej:x + 3y = 7\n 5x - y = 3").pack(anchor='w', padx=8, pady=4)
        self.txt = tk.Text(self, width=40, height=8, font=('Consolas', 11))
        self.txt.pack(padx=8, pady=4, fill='both', expand=True)
        bar = tk.Frame(self); bar.pack(fill='x')
        tk.Button(bar, text="Parsear y Rellenar", command=lambda: self._do(on_fillA, on_fillb), bg="#27ae60", fg="white").pack(side='right', padx=8, pady=8)

    def _do(self, on_fillA, on_fillb):
        try:
            A, b, vars_order = parse_linear_system(self.txt.get('1.0','end'))
            on_fillA(A)
            on_fillb([[bi] for bi in b])  # b como columna
            messagebox.showinfo("Listo", f"Variables detectadas: {', '.join(vars_order)}")
            self.destroy()
        except Exception as e:
            messagebox.showerror("Error", str(e))

# === NEW DIALOG: AlgebraicCalcDialog ============================================
import tkinter as tk
from tkinter import ttk, messagebox
from fractions import Fraction

def _fmt_frac_str(x):
    if isinstance(x, Fraction):
        return str(x.numerator) if x.denominator==1 else f"{x.numerator}/{x.denominator}"
    return str(x)

class AlgebraicCalcDialog(tk.Toplevel):
    """
    Calculadora algebraica unificada para matrices y sistemas:
      • Modo 'Expresión': evalúa expresiones tipo A+B, 2*A, T(B), inv(A)*B.
      • Modo 'Sistema': parsea ecuaciones tipo 'x+3y=7' y genera A y b.
    Callbacks:
      - getA(), getB() para leer matrices actuales
      - on_send_to_terminal(text) para imprimir resultado formateado
      - on_fillA(M) / on_fillB(M) para volcar matrices
      - on_fillb(Mcol) (opcional, para columna b en VentanaSistemas si se usara aquí)
    """
    def __init__(self, parent, getA, getB, on_send_to_terminal, on_fillA, on_fillB, on_fillb=None):
        super().__init__(parent)
        self.title("Calculadora Algebraica")
        self.geometry("640x420")
        self.getA, self.getB = getA, getB
        self.on_send, self.on_fillA, self.on_fillB = on_send_to_terminal, on_fillA, on_fillB
        self.on_fillb = on_fillb

        # Tabs: Expresión | Sistema
        nb = ttk.Notebook(self); nb.pack(fill='both', expand=True)
        frmE = tk.Frame(nb, bg='white'); nb.add(frmE, text="Expresión")
        frmS = tk.Frame(nb, bg='white'); nb.add(frmS, text="Sistema")

        # --- Expresión ---
        tk.Label(frmE, text="Expresión con A,B,+,-,*, T(), inv()", bg='white').pack(anchor='w', padx=8, pady=(8,2))
        self.errE = tk.Label(frmE, fg='red', bg='white'); self.errE.pack(anchor='w', padx=8)
        self.pad = AlgebraicKeypad(frmE, on_accept=lambda expr: self._eval_expr(expr))
        self.pad.pack(padx=8, pady=8, fill='x')
        cE = tk.Frame(frmE, bg='white'); cE.pack(fill='x', padx=8, pady=8)
        tk.Button(cE, text="Enviar a Terminal", command=lambda: self._eval_expr(self.pad.entry.get(), to_term=True)).pack(side='left')
        tk.Button(cE, text="Volcar en A", command=lambda: self._eval_expr(self.pad.entry.get(), fill='A')).pack(side='left', padx=6)
        tk.Button(cE, text="Volcar en B", command=lambda: self._eval_expr(self.pad.entry.get(), fill='B')).pack(side='left')

        # --- Sistema ---
        tk.Label(frmS, text="Escribe un sistema (una ecuación por línea):", bg='white').pack(anchor='w', padx=8, pady=(8,2))
        eg = "x + 3y = 7\n5x - y = 3"
        tk.Label(frmS, text=eg, bg='white', fg='#444').pack(anchor='w', padx=8)
        self.errS = tk.Label(frmS, fg='red', bg='white'); self.errS.pack(anchor='w', padx=8)
        self.txt = tk.Text(frmS, width=40, height=8, font=('Consolas', 11))
        self.txt.pack(padx=8, pady=6, fill='both', expand=True)
        cS = tk.Frame(frmS, bg='white'); cS.pack(fill='x', padx=8, pady=8)
        tk.Button(cS, text="Procesar → Terminal", command=lambda: self._eval_system(to_term=True)).pack(side='left')
        tk.Button(cS, text="Volcar A (coeficientes)", command=lambda: self._eval_system(fill='A')).pack(side='left', padx=6)
        tk.Button(cS, text="Volcar B (coeficientes)", command=lambda: self._eval_system(fill='B')).pack(side='left')
        tk.Button(cS, text="Volcar vector b en B", command=lambda: self._eval_system(fill='bB')).pack(side='right')

    def _format_matrix(self, M):
        if not M or not isinstance(M[0], list): return _fmt_frac_str(M)
        widths = [max(len(_fmt_frac_str(x)) for x in col) for col in zip(*M)]
        lines = []
        for r in M:
            lines.append("[ " + "  ".join(f"{_fmt_frac_str(x):>{widths[i]}}" for i,x in enumerate(r)) + " ]")
        return "\n".join(lines)

    def _eval_expr(self, expr, to_term=False, fill=None):
        try:
            env = {}
            try: env['A'] = self.getA()
            except: pass
            try: env['B'] = self.getB()
            except: pass
            M = eval_matrix_expr(expr, env)
            if to_term and callable(self.on_send):
                self.on_send(f"Expresión: {expr}\n{self._format_matrix(M)}\n")
            if fill == 'A': self.on_fillA(M)
            if fill == 'B': self.on_fillB(M)
            self.errE.config(text="")
        except Exception as e:
            self.errE.config(text=str(e))

    def _eval_system(self, to_term=False, fill=None):
        try:
            A, b, vars_order = parse_linear_system(self.txt.get('1.0','end'))
            head = f"Sistema (vars: {', '.join(vars_order)})"
            if to_term and callable(self.on_send):
                self.on_send(f"{head}\nMatriz A:\n{self._format_matrix(A)}\nVector b:\n{self._format_matrix([[bi] for bi in b])}\n")
            if fill == 'A': self.on_fillA(A)
            if fill == 'B': self.on_fillB(A)
            if fill == 'bB' and self.on_fillb is not None:
                self.on_fillb([[bi] for bi in b])
            self.errS.config(text="")
        except Exception as e:
            self.errS.config(text=str(e))
