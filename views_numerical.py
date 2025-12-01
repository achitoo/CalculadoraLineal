import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# Importamos la lógica robusta
from numerical_methods import newton_raphson, metodo_secante, metodo_biseccion, metodo_regla_falsa, _preprocesar_expresion

class VistaMetodoBase(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg="white")
        self.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.var_func = tk.StringVar()
        self.plot_timer = None
        
        top = tk.Frame(self, bg="white"); top.pack(fill=tk.X)
        self.inputs = tk.Frame(top, bg="white"); self.inputs.pack(fill=tk.X)
        self.btns = tk.Frame(top, bg="white"); self.btns.pack(fill=tk.X, pady=5)
        
        # Botones ayuda
        for v in ["sin", "cos", "tan", "ln", "^", "(", ")", "x"]:
            tk.Button(self.btns, text=v, command=lambda x=v: self._ins(x), width=3).pack(side=tk.LEFT)
        tk.Button(self.btns, text="C", command=lambda: self.var_func.set(""), width=3, bg="#ffcccc").pack(side=tk.LEFT)

        # Grafica
        self.fig = Figure(figsize=(5,4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.log = tk.Text(self, height=6, bg="#f8f9fa", font=("Consolas", 10))
        self.log.pack(fill=tk.X)
        
        self.var_func.trace_add("write", self._schedule_plot)

    def _ins(self, t): 
        if hasattr(self, 'e_func'): self.e_func.insert(tk.INSERT, t)

    def _schedule_plot(self, *a):
        if self.plot_timer: self.after_cancel(self.plot_timer)
        self.plot_timer = self.after(800, self._plot)

    def _plot(self, marker=None):
        s = self.var_func.get()
        if not s.strip(): return
        try:
            f = _preprocesar_expresion(s)
            x = np.linspace(-10, 10, 400)
            # Contexto visual (numpy)
            ctx = {"x": x, "np": np, "sin": np.sin, "cos": np.cos, "tan": np.tan, "ln": np.log, "log": np.log10, "sqrt": np.sqrt, "exp": np.exp, "pi": np.pi, "e": np.e, "abs": np.abs}
            
            with np.errstate(all='ignore'):
                y = eval(f, {"__builtins__": None}, ctx)
            
            if isinstance(y, (int, float)): y = np.full_like(x, y)
            
            self.ax.clear(); self.ax.grid(True, alpha=0.3)
            self.ax.axhline(0, color='k'); self.ax.axvline(0, color='k')
            self.ax.plot(x, y, 'b')
            if marker is not None: self.ax.plot(marker, 0, 'ro')
            
            y_ok = y[np.isfinite(y)]
            if len(y_ok)>0: 
                mn, mx = np.min(y_ok), np.max(y_ok)
                if mx-mn > 50: self.ax.set_ylim(-20, 20)
                else: self.ax.set_ylim(mn-1, mx+1)
            self.canvas.draw()
        except: pass

class VistaNewton(VistaMetodoBase):
    def __init__(self, parent):
        super().__init__(parent)
        self.x0 = tk.DoubleVar(value=1.0)
        tk.Label(self.inputs, text="f(x)=", bg="white").pack(side=tk.LEFT)
        self.e_func = tk.Entry(self.inputs, textvariable=self.var_func); self.e_func.pack(side=tk.LEFT)
        tk.Label(self.inputs, text="x0:", bg="white").pack(side=tk.LEFT)
        tk.Entry(self.inputs, textvariable=self.x0, width=5).pack(side=tk.LEFT)
        tk.Button(self.inputs, text="Calc", command=self._calc, bg="#007acc", fg="white").pack(side=tk.LEFT)

    def _calc(self):
        try:
            r, h = newton_raphson(self.var_func.get(), self.x0.get())
            self._plot(r)
            self.log.delete("1.0", tk.END); self.log.insert(tk.END, f"Raíz: {r}\n")
            for row in h: self.log.insert(tk.END, str(row)+"\n")
        except Exception as e: messagebox.showerror("Error", str(e))

class VistaSecante(VistaMetodoBase):
    def __init__(self, parent):
        super().__init__(parent)
        self.x0 = tk.DoubleVar(value=0.0); self.x1 = tk.DoubleVar(value=1.0)
        tk.Label(self.inputs, text="f(x)=", bg="white").pack(side=tk.LEFT)
        self.e_func = tk.Entry(self.inputs, textvariable=self.var_func); self.e_func.pack(side=tk.LEFT)
        tk.Entry(self.inputs, textvariable=self.x0, width=4).pack(side=tk.LEFT)
        tk.Entry(self.inputs, textvariable=self.x1, width=4).pack(side=tk.LEFT)
        tk.Button(self.inputs, text="Calc", command=self._calc, bg="#007acc", fg="white").pack(side=tk.LEFT)

    def _calc(self):
        try:
            r, h = metodo_secante(self.var_func.get(), self.x0.get(), self.x1.get())
            self._plot(r)
            self.log.delete("1.0", tk.END); self.log.insert(tk.END, f"Raíz: {r}\n")
            for row in h: self.log.insert(tk.END, str(row)+"\n")
        except Exception as e: messagebox.showerror("Error", str(e))

class VentanaBiseccion(VistaMetodoBase):
    def __init__(self, parent):
        super().__init__(parent)
        self.a = tk.DoubleVar(value=0); self.b = tk.DoubleVar(value=2)
        tk.Label(self.inputs, text="f(x)=", bg="white").pack(side=tk.LEFT)
        self.e_func = tk.Entry(self.inputs, textvariable=self.var_func); self.e_func.pack(side=tk.LEFT)
        tk.Entry(self.inputs, textvariable=self.a, width=4).pack(side=tk.LEFT)
        tk.Entry(self.inputs, textvariable=self.b, width=4).pack(side=tk.LEFT)
        tk.Button(self.inputs, text="Calc", command=self._calc, bg="#007acc", fg="white").pack(side=tk.LEFT)

    def _calc(self):
        try:
            r, h = metodo_biseccion(self.var_func.get(), self.a.get(), self.b.get())
            self._plot(r)
            self.log.delete("1.0", tk.END); self.log.insert(tk.END, f"Raíz: {r}\n")
            for row in h: self.log.insert(tk.END, str(row)+"\n")
        except Exception as e: messagebox.showerror("Error", str(e))

class VentanaReglaFalsa(VentanaBiseccion):
    def _calc(self):
        try:
            r, h = metodo_regla_falsa(self.var_func.get(), self.a.get(), self.b.get())
            self._plot(r)
            self.log.delete("1.0", tk.END); self.log.insert(tk.END, f"Raíz: {r}\n")
            for row in h: self.log.insert(tk.END, str(row)+"\n")
        except Exception as e: messagebox.showerror("Error", str(e))