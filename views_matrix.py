import tkinter as tk
from tkinter import ttk, messagebox
from fractions import Fraction
# Importamos TODO del backend
from matrix_ops import (
    sumar_matrices_dos, restar_matrices_dos, multiplicar_matrices,
    transpuesta, determinante, matriz_inversa, rango_matriz,
    regla_cramer, rref
)

class MatrixInput(tk.Frame):
    def __init__(self, parent, titulo="Matriz", filas_def=3, cols_def=3):
        super().__init__(parent, bg="white", bd=1, relief="solid")
        self.titulo = titulo
        
        h = tk.Frame(self, bg="#f8f9fa", padx=5, pady=2); h.pack(fill=tk.X)
        tk.Label(h, text=titulo, font=("bold",9), bg="#f8f9fa").pack(side=tk.LEFT)
        self.sf = tk.Spinbox(h, from_=1, to=5, width=2); self.sf.delete(0,"end"); self.sf.insert(0, filas_def); self.sf.pack(side=tk.RIGHT)
        tk.Label(h, text="x", bg="#f8f9fa").pack(side=tk.RIGHT)
        self.sc = tk.Spinbox(h, from_=1, to=5, width=2); self.sc.delete(0,"end"); self.sc.insert(0, cols_def); self.sc.pack(side=tk.RIGHT)
        tk.Button(h, text="↻", command=self._gen, bd=0).pack(side=tk.RIGHT)

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

    def get(self):
        f, c = int(self.sf.get()), int(self.sc.get())
        mat = []
        for i in range(f):
            row = []
            for j in range(c):
                v = self.ents[(i,j)].get().strip()
                # Manejo robusto de entrada vacía o decimales
                if not v: v = "0"
                try:
                    row.append(Fraction(v))
                except ValueError:
                    row.append(Fraction(float(v))) # Intenta convertir float a fraction
            mat.append(row)
        return mat

# --- ESTA ES LA CLASE QUE TE FALTA ---
class VentanaCalculadoraUniversal(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg="white")
        self.pack(fill=tk.BOTH, expand=True)
        self.ultimos_pasos = []
        
        # Selectores
        sel = tk.Frame(self, bg="#e9ecef"); sel.pack(fill=tk.X)
        self.modo = tk.StringVar(value="AB")
        tk.Radiobutton(sel, text="A y B", var=self.modo, value="AB", command=self._upd, bg="#e9ecef").pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(sel, text="Solo A", var=self.modo, value="A", command=self._upd, bg="#e9ecef").pack(side=tk.LEFT)
        tk.Radiobutton(sel, text="Solo B", var=self.modo, value="B", command=self._upd, bg="#e9ecef").pack(side=tk.LEFT)

        # Matrices
        self.fm = tk.Frame(self, bg="white"); self.fm.pack(fill=tk.X, padx=10, pady=5)
        self.mA = MatrixInput(self.fm, "Matriz A")
        self.mB = MatrixInput(self.fm, "Matriz B")

        # Operaciones
        self.fo = tk.Frame(self, bg="#f1f3f5"); self.fo.pack(fill=tk.X, padx=10)
        self.colA = tk.Frame(self.fo, bg="#f1f3f5")
        self._add(self.colA, "Ops A", ["Det", "Inv", "Transp", "Rango", "RREF(Gauss)"], "A")
        self.colAB = tk.Frame(self.fo, bg="#f1f3f5")
        self._add(self.colAB, "A y B", ["A+B", "A-B", "AxB"], "AB")
        self.colB = tk.Frame(self.fo, bg="#f1f3f5")
        self._add(self.colB, "Ops B", ["Det", "Inv", "RREF(Gauss)"], "B")

        # Consola Resultados
        f_res = tk.Frame(self, bg="white"); f_res.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        tb = tk.Frame(f_res, bg="white"); tb.pack(fill=tk.X)
        tk.Label(tb, text="Terminal de Salida:", bg="white", font=("bold",10)).pack(side=tk.LEFT)
        tk.Button(tb, text="📜 Ver Procedimiento Detallado", command=self._ver_pasos, bg="#17a2b8", fg="white").pack(side=tk.RIGHT)

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
            op_map = {"Det":"det", "Inv":"inv", "Transp":"trans", "Rango":"rango", "RREF(Gauss)":"rref", "A+B":"suma", "A-B":"resta", "AxB":"mult"}
            code = op_map.get(txt, txt)
            tk.Button(p, text=txt, bg="white", width=15, command=lambda o=code, t=tgt: self._run(o, t)).pack(pady=1)

    def _fmt_frac(self, val):
        """Convierte Fraction(5, 3) a '5/3' para visualización limpia."""
        if isinstance(val, Fraction):
            if val.denominator == 1: return str(val.numerator)
            return f"{val.numerator}/{val.denominator}"
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
                elif op=="rref": res, pasos = rref(M)
                elif op=="rango": res, pasos = rango_matriz(M)
            else: 
                if op=="suma": res, pasos = sumar_matrices_dos(A, B)
                elif op=="resta": res, pasos = restar_matrices_dos(A, B)
                elif op=="mult": res, pasos = multiplicar_matrices(A, B)

            self.txt.delete("1.0", tk.END)
            self.txt.insert(tk.END, f"> Operación: {op} ({tgt})\n")
            
            # Formato bonito en terminal
            if isinstance(res, list) and isinstance(res[0], list): # Matriz
                for row in res:
                    line = "[ " + "  ".join(f"{self._fmt_frac(x):>6}" for x in row) + " ]"
                    self.txt.insert(tk.END, line + "\n")
            else: 
                self.txt.insert(tk.END, self._fmt_frac(res))
            
            self.ultimos_pasos = pasos
            
        except Exception as e:
            self.txt.delete("1.0", tk.END)
            self.txt.insert(tk.END, f"ERROR: {str(e)}")
            self.ultimos_pasos = ["Ocurrió un error en el cálculo."]

    def _ver_pasos(self):
        if not self.ultimos_pasos: 
            messagebox.showinfo("Info", "Realiza un cálculo primero.")
            return
        win = tk.Toplevel(self); win.title("Procedimiento")
        win.geometry("600x500")
        t = tk.Text(win, font=("Consolas", 10), padx=10, pady=10)
        t.pack(fill=tk.BOTH, expand=True)
        # Limpiamos strings de pasos si vienen sucios
        for p in self.ultimos_pasos:
            t.insert(tk.END, f"{str(p)}\n{'-'*40}\n")

class VentanaSistemas(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg="white")
        self.pack(fill=tk.BOTH, expand=True, padx=20)
        tk.Label(self, text="Sistemas Ax=b", bg="white", font=("bold",12)).pack()
        self.spin = tk.Spinbox(self, from_=2, to=5, width=3, command=self._gen); self.spin.pack()
        tk.Button(self, text="Generar", command=self._gen).pack()
        self.grid = tk.Frame(self, bg="white"); self.grid.pack()
        self.entsA = {}; self.entsB = {}
        self._gen()
        
        self.pasos = []
        tk.Button(self, text="Resolver (Cramer)", command=self._solve, bg="green", fg="white").pack(pady=5)
        tk.Button(self, text="Ver Procedimiento", command=self._ver_pasos).pack()
        self.res_lbl = tk.Label(self, text="", bg="white", fg="blue", font=("bold", 11)); self.res_lbl.pack()

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

    def _solve(self):
        try:
            n = int(self.spin.get())
            A = [[Fraction(self.entsA[(i,j)].get() or 0) for j in range(n)] for i in range(n)]
            b = [Fraction(self.entsB[i].get() or 0) for i in range(n)]
            res, pasos = regla_cramer(A, b)
            self.pasos = pasos
            if res: 
                fmt_res = ", ".join([f"x{i+1}={val.numerator}/{val.denominator}" if val.denominator!=1 else f"x{i+1}={val.numerator}" for i, val in enumerate(res)])
                self.res_lbl.config(text=f"Sol: {fmt_res}")
            else: self.res_lbl.config(text="Sin sol. única")
        except: pass

    def _ver_pasos(self):
        if not self.pasos: return
        win = tk.Toplevel(self); win.title("Pasos Cramer")
        t = tk.Text(win); t.pack()
        for p in self.pasos: t.insert(tk.END, str(p)+"\n")

class VentanaVectores(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg="white")
        self.pack(fill=tk.BOTH, expand=True, padx=20)
        tk.Label(self, text="Espacios Vectoriales", bg="white", font=("bold", 12)).pack()
        
        f = tk.Frame(self, bg="white"); f.pack(pady=5)
        tk.Label(f, text="Num Vectores:", bg="white").pack(side=tk.LEFT)
        self.sv = tk.Spinbox(f, from_=2, to=5, width=3); self.sv.pack(side=tk.LEFT)
        tk.Label(f, text="Dimensión:", bg="white").pack(side=tk.LEFT)
        self.sd = tk.Spinbox(f, from_=2, to=5, width=3); self.sd.pack(side=tk.LEFT)
        tk.Button(f, text="Generar", command=self._gen).pack(side=tk.LEFT, padx=5)
        
        self.grid = tk.Frame(self, bg="white"); self.grid.pack()
        self.ents = {}
        self._gen()
        
        tk.Button(self, text="Verificar Indep.", command=self._calc, bg="blue", fg="white").pack(pady=10)
        self.lbl = tk.Label(self, text="", bg="white", font=("bold", 11)); self.lbl.pack()

    def _gen(self):
        for w in self.grid.winfo_children(): w.destroy()
        self.ents.clear()
        try: nv, dim = int(self.sv.get()), int(self.sd.get())
        except: return
        for i in range(nv):
            tk.Label(self.grid, text=f"v{i+1}:", bg="white").grid(row=i, column=0)
            for j in range(dim):
                e = tk.Entry(self.grid, width=5); e.grid(row=i, column=j+1)
                self.ents[(i,j)] = e

    def _calc(self):
        try:
            nv, dim = int(self.sv.get()), int(self.sd.get())
            M = [[Fraction(self.ents[(i,j)].get() or 0) for j in range(dim)] for i in range(nv)]
            r, _ = rango_matriz(M)
            if r == nv: self.lbl.config(text="LINEALMENTE INDEPENDIENTES", fg="green")
            else: self.lbl.config(text=f"DEPENDIENTES (Rango {r})", fg="red")
        except: pass



'''import tkinter as tk
from tkinter import ttk, messagebox
from fractions import Fraction
# Importamos TODO del backend
from matrix_ops import (
    sumar_matrices_dos, restar_matrices_dos, multiplicar_matrices,
    transpuesta, determinante, matriz_inversa, rango_matriz,
    regla_cramer, rref
)

class MatrixInput(tk.Frame):
    def __init__(self, parent, titulo="Matriz", filas_def=3, cols_def=3):
        super().__init__(parent, bg="white", bd=1, relief="solid")
        self.titulo = titulo
        
        h = tk.Frame(self, bg="#f8f9fa", padx=5, pady=2); h.pack(fill=tk.X)
        tk.Label(h, text=titulo, font=("bold",9), bg="#f8f9fa").pack(side=tk.LEFT)
        self.sf = tk.Spinbox(h, from_=1, to=5, width=2); self.sf.delete(0,"end"); self.sf.insert(0, filas_def); self.sf.pack(side=tk.RIGHT)
        tk.Label(h, text="x", bg="#f8f9fa").pack(side=tk.RIGHT)
        self.sc = tk.Spinbox(h, from_=1, to=5, width=2); self.sc.delete(0,"end"); self.sc.insert(0, cols_def); self.sc.pack(side=tk.RIGHT)
        tk.Button(h, text="↻", command=self._gen, bd=0).pack(side=tk.RIGHT)

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

    def get(self):
        f, c = int(self.sf.get()), int(self.sc.get())
        mat = []
        for i in range(f):
            row = []
            for j in range(c):
                v = self.ents[(i,j)].get().strip()
                # Manejo robusto de entrada vacía o decimales
                if not v: v = "0"
                try:
                    row.append(Fraction(v))
                except ValueError:
                    row.append(Fraction(float(v))) # Intenta convertir float a fraction
            mat.append(row)
        return mat

class VentanaCalculadoraUniversal(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg="white")
        self.pack(fill=tk.BOTH, expand=True)
        self.ultimos_pasos = []
        
        # Selectores
        sel = tk.Frame(self, bg="#e9ecef"); sel.pack(fill=tk.X)
        self.modo = tk.StringVar(value="AB")
        tk.Radiobutton(sel, text="A y B", var=self.modo, value="AB", command=self._upd, bg="#e9ecef").pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(sel, text="Solo A", var=self.modo, value="A", command=self._upd, bg="#e9ecef").pack(side=tk.LEFT)
        tk.Radiobutton(sel, text="Solo B", var=self.modo, value="B", command=self._upd, bg="#e9ecef").pack(side=tk.LEFT)

        # Matrices
        self.fm = tk.Frame(self, bg="white"); self.fm.pack(fill=tk.X, padx=10, pady=5)
        self.mA = MatrixInput(self.fm, "Matriz A")
        self.mB = MatrixInput(self.fm, "Matriz B")

        # Operaciones
        self.fo = tk.Frame(self, bg="#f1f3f5"); self.fo.pack(fill=tk.X, padx=10)
        self.colA = tk.Frame(self.fo, bg="#f1f3f5")
        self._add(self.colA, "Ops A", ["Det", "Inv", "Transp", "Rango", "RREF(Gauss)"], "A")
        self.colAB = tk.Frame(self.fo, bg="#f1f3f5")
        self._add(self.colAB, "A y B", ["A+B", "A-B", "AxB"], "AB")
        self.colB = tk.Frame(self.fo, bg="#f1f3f5")
        self._add(self.colB, "Ops B", ["Det", "Inv", "RREF(Gauss)"], "B")

        # Consola Resultados
        f_res = tk.Frame(self, bg="white"); f_res.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        tb = tk.Frame(f_res, bg="white"); tb.pack(fill=tk.X)
        tk.Label(tb, text="Terminal de Salida:", bg="white", font=("bold",10)).pack(side=tk.LEFT)
        tk.Button(tb, text="📜 Ver Procedimiento Detallado", command=self._ver_pasos, bg="#17a2b8", fg="white").pack(side=tk.RIGHT)

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
            op_map = {"Det":"det", "Inv":"inv", "Transp":"trans", "Rango":"rango", "RREF(Gauss)":"rref", "A+B":"suma", "A-B":"resta", "AxB":"mult"}
            code = op_map.get(txt, txt)
            tk.Button(p, text=txt, bg="white", width=15, command=lambda o=code, t=tgt: self._run(o, t)).pack(pady=1)

    def _fmt_frac(self, val):
        """Convierte Fraction(5, 3) a '5/3' para visualización limpia."""
        if isinstance(val, Fraction):
            if val.denominator == 1: return str(val.numerator)
            return f"{val.numerator}/{val.denominator}"
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
                elif op=="rref": res, pasos = rref(M)
                elif op=="rango": res, pasos = rango_matriz(M)
            else: 
                if op=="suma": res, pasos = sumar_matrices_dos(A, B)
                elif op=="resta": res, pasos = restar_matrices_dos(A, B)
                elif op=="mult": res, pasos = multiplicar_matrices(A, B)

            self.txt.delete("1.0", tk.END)
            self.txt.insert(tk.END, f"> Operación: {op} ({tgt})\n")
            
            # Formato bonito en terminal
            if isinstance(res, list) and isinstance(res[0], list): # Matriz
                for row in res:
                    line = "[ " + "  ".join(f"{self._fmt_frac(x):>6}" for x in row) + " ]"
                    self.txt.insert(tk.END, line + "\n")
            else: 
                self.txt.insert(tk.END, self._fmt_frac(res))
            
            self.ultimos_pasos = pasos
            
        except Exception as e:
            self.txt.delete("1.0", tk.END)
            self.txt.insert(tk.END, f"ERROR: {str(e)}")
            self.ultimos_pasos = ["Ocurrió un error en el cálculo."]

    def _ver_pasos(self):
        if not self.ultimos_pasos: 
            messagebox.showinfo("Info", "Realiza un cálculo primero.")
            return
        win = tk.Toplevel(self); win.title("Procedimiento")
        win.geometry("600x500")
        t = tk.Text(win, font=("Consolas", 10), padx=10, pady=10)
        t.pack(fill=tk.BOTH, expand=True)
        # Limpiamos strings de pasos si vienen sucios
        for p in self.ultimos_pasos:
            t.insert(tk.END, f"{str(p)}\n{'-'*40}\n")

class VentanaSistemas(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg="white")
        self.pack(fill=tk.BOTH, expand=True, padx=20)
        tk.Label(self, text="Sistemas Ax=b", bg="white", font=("bold",12)).pack()
        self.spin = tk.Spinbox(self, from_=2, to=5, width=3, command=self._gen); self.spin.pack()
        tk.Button(self, text="Generar", command=self._gen).pack()
        self.grid = tk.Frame(self, bg="white"); self.grid.pack()
        self.entsA = {}; self.entsB = {}
        self._gen()
        
        self.pasos = []
        tk.Button(self, text="Resolver (Cramer)", command=self._solve, bg="green", fg="white").pack(pady=5)
        tk.Button(self, text="Ver Procedimiento", command=self._ver_pasos).pack()
        self.res_lbl = tk.Label(self, text="", bg="white", fg="blue", font=("bold", 11)); self.res_lbl.pack()

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

    def _solve(self):
        try:
            n = int(self.spin.get())
            A = [[Fraction(self.entsA[(i,j)].get() or 0) for j in range(n)] for i in range(n)]
            b = [Fraction(self.entsB[i].get() or 0) for i in range(n)]
            res, pasos = regla_cramer(A, b)
            self.pasos = pasos
            if res: 
                fmt_res = ", ".join([f"x{i+1}={val.numerator}/{val.denominator}" if val.denominator!=1 else f"x{i+1}={val.numerator}" for i, val in enumerate(res)])
                self.res_lbl.config(text=f"Sol: {fmt_res}")
            else: self.res_lbl.config(text="Sin sol. única")
        except: pass

    def _ver_pasos(self):
        if not self.pasos: return
        win = tk.Toplevel(self); win.title("Pasos Cramer")
        t = tk.Text(win); t.pack()
        for p in self.pasos: t.insert(tk.END, str(p)+"\n")

class VentanaVectores(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg="white")
        self.pack(fill=tk.BOTH, expand=True, padx=20)
        tk.Label(self, text="Espacios Vectoriales", bg="white", font=("bold", 12)).pack()
        
        f = tk.Frame(self, bg="white"); f.pack(pady=5)
        tk.Label(f, text="Num Vectores:", bg="white").pack(side=tk.LEFT)
        self.sv = tk.Spinbox(f, from_=2, to=5, width=3); self.sv.pack(side=tk.LEFT)
        tk.Label(f, text="Dimensión:", bg="white").pack(side=tk.LEFT)
        self.sd = tk.Spinbox(f, from_=2, to=5, width=3); self.sd.pack(side=tk.LEFT)
        tk.Button(f, text="Generar", command=self._gen).pack(side=tk.LEFT, padx=5)
        
        self.grid = tk.Frame(self, bg="white"); self.grid.pack()
        self.ents = {}
        self._gen()
        
        tk.Button(self, text="Verificar Indep.", command=self._calc, bg="blue", fg="white").pack(pady=10)
        self.lbl = tk.Label(self, text="", bg="white", font=("bold", 11)); self.lbl.pack()

    def _gen(self):
        for w in self.grid.winfo_children(): w.destroy()
        self.ents.clear()
        try: nv, dim = int(self.sv.get()), int(self.sd.get())
        except: return
        for i in range(nv):
            tk.Label(self.grid, text=f"v{i+1}:", bg="white").grid(row=i, column=0)
            for j in range(dim):
                e = tk.Entry(self.grid, width=5); e.grid(row=i, column=j+1)
                self.ents[(i,j)] = e

    def _calc(self):
        try:
            nv, dim = int(self.sv.get()), int(self.sd.get())
            M = [[Fraction(self.ents[(i,j)].get() or 0) for j in range(dim)] for i in range(nv)]
            r, _ = rango_matriz(M)
            if r == nv: self.lbl.config(text="LINEALMENTE INDEPENDIENTES", fg="green")
            else: self.lbl.config(text=f"DEPENDIENTES (Rango {r})", fg="red")
        except: pass '''