import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import re  # Importación necesaria para Regex
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# Importación segura
from numerical_methods import newton_raphson, metodo_secante, metodo_biseccion, metodo_regla_falsa, _preprocesar_expresion

class VistaMetodoBase(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg="white")
        self.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.var_func = tk.StringVar()
        self.plot_timer = None
        
        # --- PANEL SUPERIOR ---
        top = tk.Frame(self, bg="white"); top.pack(fill=tk.X)
        self.inputs = tk.Frame(top, bg="white"); self.inputs.pack(fill=tk.X)
        self.btns = tk.Frame(top, bg="white"); self.btns.pack(fill=tk.X, pady=5)
        self._add_btns()
        
        # --- LABEL DE FÓRMULA ALGEBRAICA (MEJORADO) ---
        # Usamos Arial o Segoe UI para asegurar soporte de caracteres unicode
        self.lbl_formula = tk.Label(top, text="f(x) =", font=("Segoe UI", 16), bg="white", fg="#000000")
        self.lbl_formula.pack(pady=(5, 10))
        
        # --- GRÁFICA ---
        self.fig = Figure(figsize=(5,4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # --- LOG ---
        self.log = tk.Text(self, height=6, bg="#f8f9fa", font=("Consolas", 9)); self.log.pack(fill=tk.X)
        
        # Triggers
        self.var_func.trace_add("write", self._on_input_change)

    def _add_btns(self):
        # Botones de ayuda matemática
        for v in ["sin", "cos", "tan", "ln", "sqrt", "^", "(", ")", "x"]:
            tk.Button(self.btns, text=v, command=lambda x=v: self._ins(x), width=3, bg="#f0f0f0", bd=0).pack(side=tk.LEFT, padx=1)
        tk.Button(self.btns, text="C", command=self._limpiar, width=3, bg="#ffcccc", bd=0).pack(side=tk.LEFT, padx=1)

    def _ins(self, t): 
        if hasattr(self, 'e_func'): 
            # Inserta el texto y añade paréntesis si es función
            self.e_func.insert(tk.INSERT, t + ("(" if t in ["sin","cos","tan","ln","sqrt"] else ""))
            self.e_func.focus_set()

    def _limpiar(self):
        self.var_func.set("")
        if hasattr(self, 'e_func'): self.e_func.focus_set()

    def _on_input_change(self, *args):
        # 1. Actualizar el Label Algebraico INMEDIATAMENTE
        self._update_pretty_formula()
        
        # 2. Programar la gráfica con retraso (Debounce)
        if self.plot_timer: self.after_cancel(self.plot_timer)
        self.plot_timer = self.after(800, self._plot)

    def _update_pretty_formula(self):
        """Convierte la sintaxis de python a visual matemática (x^2 -> x²)."""
        txt = self.var_func.get()
        if not txt:
            self.lbl_formula.config(text="f(x) =")
            return
            
        # 1. Reemplazos básicos
        txt = txt.replace("**", "^")
        txt = txt.replace("*", "") # Quitar asteriscos de mult
        txt = txt.replace("sqrt", "√")
        txt = txt.replace("pi", "π")
        txt = txt.replace("lambda", "λ")
        
        # Tratamiento especial para exp(x) -> e^x para que luego el superscript lo agarre
        txt = re.sub(r'exp\((.*?)\)', r'e^(\1)', txt)

        # 2. Convertir exponentes a superíndices
        # Mapa de caracteres normales a superíndices
        chars_norm = "0123456789+-=()xyz"
        chars_sup  = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ˣʸᶻ"
        sup_map = str.maketrans(chars_norm, chars_sup)
        
        def to_sup(match):
            # match.group(1) es el contenido del exponente
            return match.group(1).translate(sup_map)

        # Reemplazar ^(expresion) -> ⁽ᵉˣᵖʳ⁾
        txt = re.sub(r'\^\(([0-9+\-=()xyz]+)\)', to_sup, txt)
        # Reemplazar ^char -> ᶜʰᵃʳ (ej: x^2 -> x²)
        txt = re.sub(r'\^([0-9xyz])', to_sup, txt)

        self.lbl_formula.config(text=f"f(x) = {txt}")

    def _plot(self, marker=None):
        s = self.var_func.get()
        if not s.strip(): return
        try:
            f = _preprocesar_expresion(s)
            x = np.linspace(-10, 10, 400)
            # Contexto seguro y completo
            ctx = {"x": x, "np": np, "sin": np.sin, "cos": np.cos, "tan": np.tan, 
                   "ln": np.log, "log": np.log10, "sqrt": np.sqrt, "exp": np.exp, 
                   "pi": np.pi, "e": np.e, "abs": np.abs}
            
            with np.errstate(all='ignore'):
                y = eval(f, {"__builtins__": None}, ctx)
            
            if isinstance(y, (int, float)): y = np.full_like(x, y)
            
            self.ax.clear()
            self.ax.grid(True, linestyle=':', alpha=0.6)
            self.ax.axhline(0, color='black', linewidth=1)
            self.ax.axvline(0, color='black', linewidth=1)
            
            self.ax.plot(x, y, color='#007acc', linewidth=1.5)
            if marker is not None: 
                self.ax.plot(marker, 0, 'ro', markersize=6, label='Raíz')
                self.ax.legend()
            
            # Autozoom
            y_clean = y[np.isfinite(y)]
            if len(y_clean) > 0:
                mx, mn = np.max(y_clean), np.min(y_clean)
                if mx - mn > 50: self.ax.set_ylim(-20, 20)
                else: self.ax.set_ylim(mn-2, mx+2)
            
            self.canvas.draw()
        except Exception: pass

class VistaNewton(VistaMetodoBase):
    def __init__(self, parent):
        super().__init__(parent)
        self.x0 = tk.DoubleVar(value=1.0)
        tk.Label(self.inputs, text="f(x)=", bg="white").pack(side=tk.LEFT)
        self.e_func = tk.Entry(self.inputs, textvariable=self.var_func, width=25, font=("Consolas", 11)); self.e_func.pack(side=tk.LEFT)
        tk.Label(self.inputs, text="x0:", bg="white").pack(side=tk.LEFT)
        tk.Entry(self.inputs, textvariable=self.x0, width=5).pack(side=tk.LEFT)
        tk.Button(self.inputs, text="Calcular", command=self._calc, bg="#007acc", fg="white").pack(side=tk.LEFT, padx=10)

    def _calc(self):
        try:
            r, h = newton_raphson(self.var_func.get(), self.x0.get())
            self._plot(r)
            self.log.delete("1.0", tk.END); self.log.insert(tk.END, f"Raíz: {r}\n")
            for step in h: self.log.insert(tk.END, str(step)+"\n")
        except Exception as e: messagebox.showerror("Error", f"No se pudo calcular: {e}")

class VistaSecante(VistaMetodoBase):
    def __init__(self, parent):
        super().__init__(parent)
        self.x0 = tk.DoubleVar(value=0.0); self.x1 = tk.DoubleVar(value=1.0)
        tk.Label(self.inputs, text="f(x)=", bg="white").pack(side=tk.LEFT)
        self.e_func = tk.Entry(self.inputs, textvariable=self.var_func, width=20, font=("Consolas", 11)); self.e_func.pack(side=tk.LEFT)
        tk.Label(self.inputs, text="x0:", bg="white").pack(side=tk.LEFT)
        tk.Entry(self.inputs, textvariable=self.x0, width=4).pack(side=tk.LEFT)
        tk.Label(self.inputs, text="x1:", bg="white").pack(side=tk.LEFT)
        tk.Entry(self.inputs, textvariable=self.x1, width=4).pack(side=tk.LEFT)
        tk.Button(self.inputs, text="Calcular", command=self._calc, bg="#007acc", fg="white").pack(side=tk.LEFT, padx=10)

    def _calc(self):
        try:
            r, h = metodo_secante(self.var_func.get(), self.x0.get(), self.x1.get())
            self._plot(r)
            self.log.delete("1.0", tk.END); self.log.insert(tk.END, f"Raíz: {r}\n")
            for step in h: self.log.insert(tk.END, str(step)+"\n")
        except Exception as e: messagebox.showerror("Error", str(e))

class VentanaBiseccion(VistaMetodoBase):
    def __init__(self, parent):
        super().__init__(parent)
        self.a = tk.DoubleVar(value=0); self.b = tk.DoubleVar(value=2)
        tk.Label(self.inputs, text="f(x)=", bg="white").pack(side=tk.LEFT)
        self.e_func = tk.Entry(self.inputs, textvariable=self.var_func, width=20, font=("Consolas", 11)); self.e_func.pack(side=tk.LEFT)
        tk.Label(self.inputs, text="[a, b]", bg="white").pack(side=tk.LEFT)
        tk.Entry(self.inputs, textvariable=self.a, width=4).pack(side=tk.LEFT)
        tk.Entry(self.inputs, textvariable=self.b, width=4).pack(side=tk.LEFT)
        tk.Button(self.inputs, text="Bisección", command=self._calc, bg="#007acc", fg="white").pack(side=tk.LEFT, padx=10)

    def _calc(self):
        try:
            from fractions import Fraction
            r, h = metodo_biseccion(self.var_func.get(), float(self.a.get()), float(self.b.get())) # Usamos floats para numerical
            self._plot(float(r))
            self.log.delete("1.0", tk.END); self.log.insert(tk.END, f"Raíz: {float(r)}\n")
            for step in h: self.log.insert(tk.END, str(step)+"\n")
        except Exception as e: messagebox.showerror("Error", str(e))

class VentanaReglaFalsa(VentanaBiseccion):
    def _calc(self):
        try:
            from fractions import Fraction
            r, h = metodo_regla_falsa(self.var_func.get(), float(self.a.get()), float(self.b.get()))
            self._plot(float(r))
            self.log.delete("1.0", tk.END); self.log.insert(tk.END, f"Raíz: {float(r)}\n")
            for step in h: self.log.insert(tk.END, str(step)+"\n")
        except Exception as e: messagebox.showerror("Error", str(e))