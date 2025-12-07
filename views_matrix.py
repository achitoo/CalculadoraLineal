import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from fractions import Fraction
import random

from matrix_ops import (
    sumar_matrices_dos, restar_matrices_dos, multiplicar_matrices,
    transpuesta, determinante, matriz_inversa, rango_matriz,
    regla_cramer, resolver_gauss, resolver_gauss_jordan, rref_con_pasos
)
from algebraic_fill import parsear_matriz_texto, parsear_sistema_ecuaciones

class MatrixInput(tk.Frame):
    """Componente Grid con herramientas avanzadas de generación."""
    def __init__(self, parent, titulo="Matriz", filas_def=3, cols_def=3):
        super().__init__(parent, bg="white", bd=1, relief="solid")
        self.titulo = titulo
        
        # Header
        h = tk.Frame(self, bg="#f8f9fa", padx=5, pady=2); h.pack(fill=tk.X)
        tk.Label(h, text=titulo, font=("bold",9), bg="#f8f9fa").pack(side=tk.LEFT)
        
        # --- BARRA DE HERRAMIENTAS ---
        
        # Botón Pegar
        tk.Button(h, text="📋", command=self._importar_texto, bd=0, bg="#e2e6ea", width=3, cursor="hand2").pack(side=tk.RIGHT, padx=1)
        
        # Menú Generar (Identidades y Especiales)
        mb_gen = tk.Menubutton(h, text="⚡ Generar", bd=0, bg="#d1ecf1", cursor="hand2", font=("Segoe UI", 8))
        menu_gen = tk.Menu(mb_gen, tearoff=0)
        mb_gen.config(menu=menu_gen)
        
        menu_gen.add_command(label="Matriz Identidad (I)", command=self._hacer_identidad)
        menu_gen.add_command(label="Matriz Nula (0)", command=self._limpiar_ceros)
        menu_gen.add_command(label="Matriz Escalar (k·I)...", command=self._hacer_escalar)
        menu_gen.add_separator()
        menu_gen.add_command(label="Aleatoria (Enteros)", command=self._hacer_random)
        menu_gen.add_command(label="Aleatoria (Simétrica)", command=self._hacer_simetrica)
        menu_gen.add_command(label="Aleatoria (Diagonal)", command=self._hacer_diagonal)
        
        mb_gen.pack(side=tk.RIGHT, padx=1)

        # Dimensiones
        self.sf = tk.Spinbox(h, from_=1, to=8, width=2); self.sf.delete(0,"end"); self.sf.insert(0, filas_def); self.sf.pack(side=tk.RIGHT)
        tk.Label(h, text="x", bg="#f8f9fa").pack(side=tk.RIGHT)
        self.sc = tk.Spinbox(h, from_=1, to=8, width=2); self.sc.delete(0,"end"); self.sc.insert(0, cols_def); self.sc.pack(side=tk.RIGHT)
        tk.Button(h, text="↻", command=self._gen, bd=0, bg="#e9ecef").pack(side=tk.RIGHT, padx=2)

        self.grid = tk.Frame(self, bg="white", padx=5, pady=5); self.grid.pack()
        self.ents = {}; self._gen()

    def _gen(self):
        for w in self.grid.winfo_children(): w.destroy()
        self.ents.clear()
        try: f, c = int(self.sf.get()), int(self.sc.get())
        except: return
        for i in range(f):
            for j in range(c):
                e = tk.Entry(self.grid, width=5, justify="center", bg="#f8f9fa")
                e.grid(row=i, column=j, padx=1, pady=1)
                self.ents[(i,j)] = e

    # --- MÉTODOS DE LLENADO ---
    
    def _limpiar_ceros(self):
        for e in self.ents.values(): e.delete(0,tk.END); e.insert(0,"0")

    def _hacer_identidad(self):
        self._llenar_diagonal(val_diag="1", val_resto="0")

    def _hacer_escalar(self):
        k = simpledialog.askstring("Matriz Escalar", "Ingresa el valor escalar (k):", parent=self)
        if k is not None:
            self._llenar_diagonal(val_diag=k, val_resto="0")

    def _hacer_random(self):
        try:
            f, c = int(self.sf.get()), int(self.sc.get())
            for i in range(f):
                for j in range(c):
                    val = str(random.randint(-9, 9))
                    self.ents[(i,j)].delete(0, tk.END)
                    self.ents[(i,j)].insert(0, val)
        except: pass

    def _hacer_simetrica(self):
        try:
            f, c = int(self.sf.get()), int(self.sc.get())
            if f != c:
                messagebox.showwarning("Aviso", "Para ser simétrica debe ser cuadrada.")
                return
            # Llenar triangulo superior y copiar al inferior
            for i in range(f):
                for j in range(i, c):
                    val = str(random.randint(-9, 9))
                    # Asignar (i, j)
                    self.ents[(i,j)].delete(0, tk.END); self.ents[(i,j)].insert(0, val)
                    # Asignar (j, i)
                    self.ents[(j,i)].delete(0, tk.END); self.ents[(j,i)].insert(0, val)
        except: pass

    def _hacer_diagonal(self):
        try:
            f, c = int(self.sf.get()), int(self.sc.get())
            for i in range(f):
                for j in range(c):
                    self.ents[(i,j)].delete(0, tk.END)
                    if i == j:
                        self.ents[(i,j)].insert(0, str(random.randint(-9, 9)))
                    else:
                        self.ents[(i,j)].insert(0, "0")
        except: pass

    def _llenar_diagonal(self, val_diag, val_resto):
        try:
            f, c = int(self.sf.get()), int(self.sc.get())
            for i in range(f):
                for j in range(c):
                    self.ents[(i,j)].delete(0, tk.END)
                    self.ents[(i,j)].insert(0, val_diag if i == j else val_resto)
        except: pass

    def _importar_texto(self):
        win = tk.Toplevel(self); win.title("Pegar")
        tk.Label(win, text="Pega aquí:").pack()
        txt = tk.Text(win, height=5, width=30); txt.pack(padx=5)
        def procesar():
            raw = txt.get("1.0", tk.END)
            d = parsear_matriz_texto(raw)
            if d:
                self.sf.delete(0,"end"); self.sf.insert(0, len(d))
                self.sc.delete(0,"end"); self.sc.insert(0, len(d[0]))
                self._gen()
                for i,r in enumerate(d):
                    for j,v in enumerate(r):
                        if (i,j) in self.ents: self.ents[(i,j)].delete(0,tk.END); self.ents[(i,j)].insert(0,str(v))
            win.destroy()
        tk.Button(win, text="Cargar", command=procesar).pack(pady=5)

    def get(self):
        f, c = int(self.sf.get()), int(self.sc.get())
        mat = []
        for i in range(f):
            row = []
            for j in range(c):
                v = self.ents[(i,j)].get().strip()
                try: row.append(Fraction(v if v else "0"))
                except: row.append(Fraction(0))
            mat.append(row)
        return mat

# --- VISTA 1: CALCULADORA UNIVERSAL ---
class VentanaCalculadoraUniversal(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg="white")
        self.pack(fill=tk.BOTH, expand=True)
        self.ultimos_pasos = []
        
        sel = tk.Frame(self, bg="#e9ecef"); sel.pack(fill=tk.X)
        self.modo = tk.StringVar(value="AB")
        style = {"bg": "#e9ecef", "activebackground": "#dee2e6", "cursor": "hand2"}
        tk.Radiobutton(sel, text="A y B", var=self.modo, value="AB", command=self._upd, **style).pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(sel, text="Solo A", var=self.modo, value="A", command=self._upd, **style).pack(side=tk.LEFT)
        tk.Radiobutton(sel, text="Solo B", var=self.modo, value="B", command=self._upd, **style).pack(side=tk.LEFT)

        self.fm = tk.Frame(self, bg="white"); self.fm.pack(fill=tk.X, padx=10, pady=5)
        self.mA = MatrixInput(self.fm, "Matriz A")
        self.mB = MatrixInput(self.fm, "Matriz B")

        self.fo = tk.Frame(self, bg="#f1f3f5"); self.fo.pack(fill=tk.X, padx=10)
        self.colA = tk.Frame(self.fo, bg="#f1f3f5")
        self._add(self.colA, "Ops A", ["Det", "Inv", "Transp", "Rango"], "A")
        self.colAB = tk.Frame(self.fo, bg="#f1f3f5")
        self._add(self.colAB, "A y B", ["A+B", "A-B", "AxB"], "AB")
        self.colB = tk.Frame(self.fo, bg="#f1f3f5")
        self._add(self.colB, "Ops B", ["Det", "Inv"], "B")

        f_res = tk.Frame(self, bg="white"); f_res.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        tb = tk.Frame(f_res, bg="white"); tb.pack(fill=tk.X)
        tk.Button(tb, text="📜 Ver Procedimiento", command=self._ver_pasos, bg="#17a2b8", fg="white").pack(side=tk.RIGHT)
        self.txt = tk.Text(f_res, bg="#212529", fg="#00ff00", height=8, font=("Consolas", 10))
        self.txt.pack(fill=tk.BOTH, expand=True)
        self._upd()

    def _upd(self):
        m = self.modo.get()
        self.mA.pack_forget(); self.mB.pack_forget()
        self.colA.pack_forget(); self.colAB.pack_forget(); self.colB.pack_forget()
        if m=="A": self.mA.pack(fill=tk.X); self.colA.pack(fill=tk.X)
        elif m=="B": self.mB.pack(fill=tk.X); self.colB.pack(fill=tk.X)
        else: 
            self.mA.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
            self.mB.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
            self.colAB.pack(fill=tk.X)

    def _add(self, p, t, b, tgt):
        tk.Label(p, text=t, bg="#f1f3f5", font=("bold")).pack()
        for txt in b:
            op_map = {"Det":"det", "Inv":"inv", "Transp":"trans", "Rango":"rango", "A+B":"suma", "A-B":"resta", "AxB":"mult"}
            code = op_map.get(txt, txt)
            tk.Button(p, text=txt, bg="white", width=15, command=lambda o=code, t=tgt: self._run(o, t)).pack(pady=1)

    def _fmt(self, val):
        if isinstance(val, Fraction): return str(val.numerator) if val.denominator==1 else f"{val.numerator}/{val.denominator}"
        return str(val)

    def _run(self, op, tgt):
        try:
            res, pasos = None, []
            A = self.mA.get() if tgt in ["A", "AB"] else None
            B = self.mB.get() if tgt in ["B", "AB"] else None
            
            if tgt != "AB": 
                M = A if tgt=="A" else B
                if op=="det": res, pasos = determinante(M)
                elif op=="inv": res, pasos = matriz_inversa(M)
                elif op=="trans": res, pasos = transpuesta(M)
                elif op=="rango": res, pasos = rango_matriz(M)
            else: 
                if op=="suma": res, pasos = sumar_matrices_dos(A, B)
                elif op=="resta": res, pasos = restar_matrices_dos(A, B)
                elif op=="mult": res, pasos = multiplicar_matrices(A, B)

            self.txt.delete("1.0", tk.END)
            self.txt.insert(tk.END, f"> Operación: {op} ({tgt})\n")
            if isinstance(res, list) and isinstance(res[0], list):
                for row in res: self.txt.insert(tk.END, "[ " + "  ".join(f"{self._fmt(x):>6}" for x in row) + " ]\n")
            else: self.txt.insert(tk.END, self._fmt(res))
            self.ultimos_pasos = pasos
        except Exception as e:
            self.txt.delete("1.0", tk.END); self.txt.insert(tk.END, f"ERROR: {str(e)}")

    def _ver_pasos(self):
        if not self.ultimos_pasos: return
        win = tk.Toplevel(self); win.title("Procedimiento")
        t = tk.Text(win, font=("Consolas", 10), padx=10, pady=10)
        t.pack(fill=tk.BOTH, expand=True)
        for p in self.ultimos_pasos: t.insert(tk.END, f"{str(p)}\n{'-'*40}\n")

# --- VISTA 2: CRAMER ---
class VentanaSistemas(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg="white")
        self.pack(fill=tk.BOTH, expand=True, padx=20)
        tk.Label(self, text="Sistemas Ax=b (Cramer)", bg="white", font=("bold",14)).pack(pady=10)
        
        self.nb = ttk.Notebook(self); self.nb.pack(fill=tk.BOTH, expand=True)
        self.tab_vis = tk.Frame(self.nb, bg="white"); self.nb.add(self.tab_vis, text="Visual")
        self.tab_txt = tk.Frame(self.nb, bg="white"); self.nb.add(self.tab_txt, text="Texto")
        
        # Visual
        f = tk.Frame(self.tab_vis, bg="white"); f.pack(pady=5)
        tk.Label(f, text="N:", bg="white").pack(side=tk.LEFT)
        self.spin = tk.Spinbox(f, from_=2, to=5, width=3, command=self._gen); self.spin.pack(side=tk.LEFT)
        tk.Button(f, text="Generar", command=self._gen).pack(side=tk.LEFT)
        self.grid = tk.Frame(self.tab_vis, bg="white"); self.grid.pack()
        self.entsA = {}; self.entsB = {}
        tk.Button(self.tab_vis, text="Resolver (Cramer)", command=self._solve_vis, bg="green", fg="white").pack(pady=10)
        
        # Texto
        tk.Label(self.tab_txt, text="Ej: 2x+y=10", bg="white").pack()
        self.txt_ec = tk.Text(self.tab_txt, height=8); self.txt_ec.pack(fill=tk.X)
        tk.Button(self.tab_txt, text="Interpretar y Resolver", command=self._solve_txt, bg="#007acc", fg="white").pack(pady=5)

        self.res_lbl = tk.Label(self, text="", bg="white", fg="blue", font=("bold",11)); self.res_lbl.pack(pady=5)
        self.pasos = []
        tk.Button(self, text="Ver Procedimiento", command=self._ver_pasos).pack()
        self._gen()

    def _gen(self):
        for w in self.grid.winfo_children(): w.destroy()
        self.entsA.clear(); self.entsB.clear()
        try: n = int(self.spin.get())
        except: return
        for i in range(n):
            for j in range(n):
                e = tk.Entry(self.grid, width=5); e.grid(row=i, column=j)
                self.entsA[(i,j)] = e
            tk.Label(self.grid, text="=", bg="white").grid(row=i, column=n)
            eb = tk.Entry(self.grid, width=5); eb.grid(row=i, column=n+1)
            self.entsB[i] = eb

    def _solve_vis(self):
        try:
            n = int(self.spin.get())
            A = [[Fraction(self.entsA[(i,j)].get() or 0) for j in range(n)] for i in range(n)]
            b = [Fraction(self.entsB[i].get() or 0) for i in range(n)]
            self._exec(A, b)
        except Exception as e: messagebox.showerror("Error", str(e))

    def _solve_txt(self):
        try:
            A, b, _ = parsear_sistema_ecuaciones(self.txt_ec.get("1.0", tk.END))
            if A: self._exec(A, b)
        except Exception as e: messagebox.showerror("Error", str(e))

    def _exec(self, A, b):
        res, pasos = regla_cramer(A, b)
        self.pasos = pasos
        if res:
            fmt = ", ".join([f"x{i+1}={val.numerator}/{val.denominator}" if val.denominator!=1 else f"x{i+1}={val.numerator}" for i, val in enumerate(res)])
            self.res_lbl.config(text=f"Solución: {fmt}")
        else: self.res_lbl.config(text="Sin solución única")

    def _ver_pasos(self):
        if not self.pasos: return
        win = tk.Toplevel(self); win.title("Pasos Cramer")
        t = tk.Text(win); t.pack(fill=tk.BOTH, expand=True)
        for p in self.pasos: t.insert(tk.END, str(p)+"\n")

# --- VISTA 3: GAUSS / GAUSS-JORDAN ---
class VentanaGauss(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg="white")
        self.pack(fill=tk.BOTH, expand=True, padx=20)
        tk.Label(self, text="Sistemas Ax=b (Gauss / Gauss-Jordan)", bg="white", font=("bold",14)).pack(pady=10)
        
        self.nb = ttk.Notebook(self); self.nb.pack(fill=tk.BOTH, expand=True)
        self.tab_vis = tk.Frame(self.nb, bg="white"); self.nb.add(self.tab_vis, text="Cuadrícula")
        self.tab_txt = tk.Frame(self.nb, bg="white"); self.nb.add(self.tab_txt, text="Texto")
        
        # Visual
        f = tk.Frame(self.tab_vis, bg="white"); f.pack(pady=5)
        tk.Label(f, text="N:", bg="white").pack(side=tk.LEFT)
        self.spin = tk.Spinbox(f, from_=2, to=5, width=3, command=self._gen); self.spin.pack(side=tk.LEFT)
        tk.Button(f, text="Generar", command=self._gen).pack(side=tk.LEFT)
        
        self.grid = tk.Frame(self.tab_vis, bg="white"); self.grid.pack()
        self.entsA = {}; self.entsB = {}
        
        f_btns = tk.Frame(self.tab_vis, bg="white"); f_btns.pack(pady=10)
        tk.Button(f_btns, text="Resolver por Gauss", command=lambda: self._solve_vis("gauss"), bg="#17a2b8", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(f_btns, text="Resolver por Gauss-Jordan", command=lambda: self._solve_vis("rref"), bg="#28a745", fg="white").pack(side=tk.LEFT, padx=5)

        # Texto
        tk.Label(self.tab_txt, text="Ej: x+y=3", bg="white").pack()
        self.txt_ec = tk.Text(self.tab_txt, height=8); self.txt_ec.pack(fill=tk.X)
        f_btns2 = tk.Frame(self.tab_txt, bg="white"); f_btns2.pack(pady=10)
        tk.Button(f_btns2, text="Gauss", command=lambda: self._solve_txt("gauss"), bg="#17a2b8", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(f_btns2, text="Gauss-Jordan", command=lambda: self._solve_txt("rref"), bg="#28a745", fg="white").pack(side=tk.LEFT, padx=5)

        self.res_lbl = tk.Label(self, text="", bg="white", fg="blue", font=("bold",11)); self.res_lbl.pack(pady=5)
        self.pasos = []
        tk.Button(self, text="Ver Procedimiento", command=self._ver_pasos).pack()
        self._gen()

    def _gen(self):
        for w in self.grid.winfo_children(): w.destroy()
        self.entsA.clear(); self.entsB.clear()
        try: n = int(self.spin.get())
        except: return
        for i in range(n):
            for j in range(n):
                e = tk.Entry(self.grid, width=5); e.grid(row=i, column=j)
                self.entsA[(i,j)] = e
            tk.Label(self.grid, text="=", bg="white").grid(row=i, column=n)
            eb = tk.Entry(self.grid, width=5); eb.grid(row=i, column=n+1)
            self.entsB[i] = eb

    def _solve_vis(self, method):
        try:
            n = int(self.spin.get())
            A = [[Fraction(self.entsA[(i,j)].get() or 0) for j in range(n)] for i in range(n)]
            b = [Fraction(self.entsB[i].get() or 0) for i in range(n)]
            self._exec(A, b, method)
        except Exception as e: messagebox.showerror("Error", str(e))

    def _solve_txt(self, method):
        try:
            A, b, _ = parsear_sistema_ecuaciones(self.txt_ec.get("1.0", tk.END))
            if A: self._exec(A, b, method)
        except Exception as e: messagebox.showerror("Error", str(e))

    def _exec(self, A, b, method):
        if method == "gauss": res, pasos = resolver_gauss(A, b)
        else: res, pasos = resolver_gauss_jordan(A, b)
        
        self.pasos = pasos
        if res:
            fmt = ", ".join([f"x{i+1}={val.numerator}/{val.denominator}" if val.denominator!=1 else f"x{i+1}={val.numerator}" for i, val in enumerate(res)])
            self.res_lbl.config(text=f"Solución: {fmt}")
        else: self.res_lbl.config(text="Sin solución única")

    def _ver_pasos(self):
        if not self.pasos: return
        win = tk.Toplevel(self); win.title("Pasos")
        t = tk.Text(win); t.pack(fill=tk.BOTH, expand=True)
        for p in self.pasos: t.insert(tk.END, str(p)+"\n")

class VentanaVectores(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg="white")
        self.pack(fill=tk.BOTH, expand=True, padx=20)
        tk.Label(self, text="Espacios Vectoriales", bg="white", font=("bold", 12)).pack()
        self.v_input = MatrixInput(self, "Vectores", 3, 3); self.v_input.pack(fill=tk.X)
        tk.Button(self, text="Verificar Indep.", command=self._calc, bg="blue", fg="white").pack(pady=10)
        self.lbl = tk.Label(self, text="", bg="white", font=("bold", 11)); self.lbl.pack()

    def _calc(self):
        try:
            M = self.v_input.get()
            r, _ = rango_matriz(M)
            if r == len(M): self.lbl.config(text="LINEALMENTE INDEPENDIENTES", fg="green")
            else: self.lbl.config(text=f"DEPENDIENTES (Rango {r})", fg="red")
        except: pass