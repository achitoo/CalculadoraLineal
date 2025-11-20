import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable, List, Optional, Sequence, Union
from fractions import Fraction
import re
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from numerical_methods import newton_raphson, _preprocesar_expresion

def _normalizar_expresion(expr: str) -> str:
    # Limpieza básica
    expr = (expr or "").strip()
    expr = expr.replace(",", ".")            # 0,5 -> 0.5
    expr = expr.replace("×", "*").replace("·", "*")
    expr = expr.replace("^", "**")           # ^ -> **

    # Inserciones de multiplicación implícita
    expr = re.sub(r'(?<=\d)\s*(?=x\b)', '*', expr)            # 2x -> 2*x / 0.5x -> 0.5*x
    expr = re.sub(r'(?<=\d)\s*(?=\()', '*', expr)             # 2(x+1) -> 2*(x+1)
    expr = re.sub(r'\)\s*(?=\()', ')*(', expr)                # )( -> )*(
    expr = re.sub(r'\)\s*(?=(\d|x|\bpi\b|\be\b))', ')*', expr)# )x, )2, )pi -> )*x ...
    expr = re.sub(r'(?<=x)\s*(?=\()', '*', expr)              # x( -> x*(
    expr = re.sub(r'(?<=\b(?:pi|e))\s*(?=x\b)', '*', expr)    # pi x -> pi*x
    expr = re.sub(r'\b(sin|cos|tan|log|log10|exp|sqrt|asin|acos|atan|sinh|cosh|tanh)\s*x\b',
                r'\1(x)', expr)                             # sin x -> sin(x)

    return expr


from matrix_ops import (
    formatear_numero,
    multiplicar_cadena,
    multiplicar_matrices,
    restar_matrices_secuencial,
    sumar_matrices_lista,
    rango_matriz,
    transpuesta,
    es_multiplicable,
    vector_columna,
    vector_fila,
    identidad,
    inversa_gauss_jordan,
    caracterizaciones_invertible,
    determinante,
    regla_cramer,
    RegistroEntrada,
    Registro,
)
from simple_calculator import SimpleCalculator, format_fraction_unicode
from numerical_methods import (
    metodo_biseccion,
    metodo_regla_falsa,
    secante_falsaPosicion,
    RegistroBiseccion,
    _crear_evaluador_numpy,
)
import numpy as np
import matplotlib
matplotlib.use("TkAgg") # Enlace entre Matplotlib y Tkinter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk

def convertir_numero(texto: str) -> Fraction:
    """Convierte una cadena en fraccion."""
    texto = (texto or "").strip()
    if texto == "":
        return Fraction(0)
    
    texto = texto.replace(",", ".")
    
    if "/" in texto:
        try:
            num, den = texto.split("/")
            return Fraction(int(num), int(den))
        except (ValueError, ZeroDivisionError) as e:
            raise ValueError(f"Fracción inválida: '{texto}'") from e
    else:
        try:
            return Fraction(texto)
        except ValueError as e:
            raise ValueError(f"Número o decimal inválido: '{texto}'") from e


NumberDisplay = Union[Fraction, float, int]


def formatear_valor_ui(valor: NumberDisplay) -> str:
    """Devuelve una representación amigable (fracciones en formato unicode)."""
    if isinstance(valor, Fraction):
        return format_fraction_unicode(valor)
    if isinstance(valor, float):
        texto = f"{valor:.10g}"
        if "." in texto:
            texto = texto.rstrip("0").rstrip(".")
        return texto or "0"
    return str(valor)

# --- CLASE VISTA NEWTON-RAPHSON MEJORADA ---
class VistaNewton(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # --- Variables ---
        self.var_funcion = tk.StringVar()
        self.var_x0 = tk.DoubleVar(value=1.0)
        self.var_tol = tk.DoubleVar(value=1e-6)
        self._plot_timer = None 

        # --- Interfaz ---
        self._crear_controles()
        self._crear_area_grafica()
        
        # --- Eventos ---
        # Detectar cambios para graficar automáticamente
        self.var_funcion.trace_add("write", self._on_funcion_change)
        self.var_x0.trace_add("write", self._on_funcion_change)

    def _crear_controles(self):
        # Contenedor principal de controles
        frame_controles = tk.Frame(self)
        frame_controles.pack(side=tk.TOP, fill=tk.X, pady=5)

        # --- Fila 1: Entradas de Texto ---
        frame_fila1 = tk.Frame(frame_controles)
        frame_fila1.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        tk.Label(frame_fila1, text="f(x) =").pack(side=tk.LEFT)
        
        # Entry guardado en self para poder insertar texto desde botones
        self.entry_func = tk.Entry(frame_fila1, textvariable=self.var_funcion, width=35, font=("Consolas", 11))
        self.entry_func.pack(side=tk.LEFT, padx=5)
        # Al presionar Enter, calcula la raíz
        self.entry_func.bind("<Return>", lambda e: self._calcular()) 
        
        tk.Label(frame_fila1, text="X0:").pack(side=tk.LEFT)
        tk.Entry(frame_fila1, textvariable=self.var_x0, width=8).pack(side=tk.LEFT, padx=5)

        btn_calc = tk.Button(frame_fila1, text="Calcular Raíz", command=self._calcular, 
                             bg="#007acc", fg="white", font=("Arial", 9, "bold"))
        btn_calc.pack(side=tk.LEFT, padx=10)

        # --- Fila 2: Botones Matemáticos ---
        frame_botones = tk.Frame(frame_controles)
        frame_botones.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        # Lista de tuplas: (Texto Botón, Texto a Insertar)
        botones = [
            ("sin", "sin("), ("cos", "cos("), ("tan", "tan("),
            ("ln", "ln("), ("log", "log10("), ("e", "e"), 
            ("π", "pi"), ("√", "sqrt("), ("^", "^"), 
            ("(", "("), (")", ")"), ("Borrar", "CLEAR")
        ]

        for txt, val in botones:
            if val == "CLEAR":
                cmd = self._limpiar_funcion
                bg_color = "#ffcccc" # Rojo claro para limpiar
                width_btn = 6
            else:
                cmd = lambda v=val: self._insertar_texto(v)
                bg_color = "#f0f0f0"
                width_btn = 4

            tk.Button(frame_botones, text=txt, command=cmd, width=width_btn, bg=bg_color).pack(side=tk.LEFT, padx=1)

    def _insertar_texto(self, texto):
        """Inserta texto en la posición del cursor y devuelve el foco."""
        try:
            idx = self.entry_func.index(tk.INSERT)
            self.entry_func.insert(idx, texto)
            self.entry_func.focus_set()
        except:
            # Fallback si no tiene foco
            current = self.var_funcion.get()
            self.var_funcion.set(current + texto)

    def _limpiar_funcion(self):
        self.var_funcion.set("")
        self.entry_func.focus_set()

    def _crear_area_grafica(self):
        # Figura de Matplotlib
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self._configurar_ejes()

        # Canvas de Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Area de resultados
        self.txt_resultados = tk.Text(self, height=8, font=("Consolas", 9))
        self.txt_resultados.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

    def _configurar_ejes(self):
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("f(x)")
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.ax.axhline(0, color='black', linewidth=1)
        self.ax.axvline(0, color='black', linewidth=1)

    def _on_funcion_change(self, *args):
        # Debounce: Esperar 500ms de inactividad antes de graficar
        if self._plot_timer is not None:
            self.after_cancel(self._plot_timer)
        self._plot_timer = self.after(500, self._actualizar_grafica)

    def _actualizar_grafica(self):
        funcion_str = self.var_funcion.get()
        if not funcion_str.strip():
            return

        # ESTO ES LO QUE BORRABA TUS RESULTADOS
        self.txt_resultados.delete(1.0, tk.END)

        try:
            # 1. Preprocesamiento
            # Primero un reemplazo visual básico
            func_py = funcion_str.replace('^', '**')
            # Intentamos usar el preprocesador robusto del módulo numerical_methods
            try:
                func_py = _preprocesar_expresion(funcion_str)
            except Exception:
                pass

            # 2. Obtener X0 para centrar gráfica
            try:
                val_x0 = self.var_x0.get()
            except:
                val_x0 = 0.0
            
            # 3. Generar datos X
            rango = 10 
            x = np.linspace(val_x0 - rango, val_x0 + rango, 400)
            
            # 4. Contexto Numpy (Mapeo de funciones)
            contexto_np = {
                "x": x, "np": np,
                "sin": np.sin, "cos": np.cos, "tan": np.tan,
                "sqrt": np.sqrt, "exp": np.exp, 
                "log": np.log, "ln": np.log, "log10": np.log10,
                "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
                "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
                "pi": np.pi, "e": np.e, "abs": np.abs
            }

            # 5. Evaluar
            y = eval(func_py, {"__builtins__": None}, contexto_np)

            # Convertir escalar a array si es necesario (ej: f(x) = 5)
            if isinstance(y, (int, float, np.number)):
                y = np.full_like(x, float(y))
            
            # 6. Dibujar
            self.ax.clear()
            self._configurar_ejes()
            self.ax.plot(x, y, color='blue', linewidth=1.5, label='f(x)')
            self.ax.plot(val_x0, 0, 'ro', markersize=8, label='X0')
            
            # 7. Auto-Zoom inteligente (evitar asíntotas gigantes)
            y_validos = y[np.isfinite(y)]
            if len(y_validos) > 0:
                ymin, ymax = np.min(y_validos), np.max(y_validos)
                # Si el rango es gigante (posible asíntota), cortamos
                if ymax - ymin > 50:
                    self.ax.set_ylim(-25, 25)
                else:
                    margen = (ymax - ymin) * 0.1 if ymax != ymin else 1.0
                    self.ax.set_ylim(ymin - margen, ymax + margen)
            
            self.ax.legend()
            self.canvas.draw()

        except Exception as e:
            # Mostrar error de sintaxis discretamente en el log
            self.txt_resultados.insert(tk.END, f"Esperando ecuación válida... ({str(e)})\n")

    def _calcular(self):
        f_str = self.var_funcion.get()
        try:
            x0 = self.var_x0.get()
            tol = self.var_tol.get()
            
            # Llamada al método numérico real
            raiz, registro = newton_raphson(f_str, x0, tol)
            
            # 1. PRIMERO ACTUALIZAMOS LA GRÁFICA
            self._actualizar_grafica()
            self.ax.plot(raiz, 0, 'go', markersize=8, label='Raíz')
            self.ax.legend()
            self.canvas.draw()
            
            # 2. LUEGO ESCRIBIMOS LOS RESULTADOS (Así no se borran)
            self.txt_resultados.delete(1.0, tk.END)
            self.txt_resultados.insert(tk.END, f"Raíz encontrada: {raiz}\n\n")
            self.txt_resultados.insert(tk.END, f"{'Iter':<5} | {'xi':<12} | {'f(xi)':<12} | {'Error':<12}\n")
            self.txt_resultados.insert(tk.END, "-"*60 + "\n")
            
            for fila in registro:
                err_val = fila['error']
                # Formateo seguro del error
                err_str = f"{err_val:.2e}" if isinstance(err_val, (int, float)) else str(err_val)
                
                self.txt_resultados.insert(tk.END, 
                    f"{fila['iter']:<5} | {fila['xi']:.8f} | {fila['f(xi)']:.2e} | {err_str}\n"
                )
            
        except Exception as e:
            tk.messagebox.showerror("Error de Cálculo", f"No se pudo calcular: {str(e)}")


class VentanaIndependencia(ttk.Frame):
    """Vista para verificar independencia lineal de vectores (columnas)."""

    def __init__(self, maestro):
        super().__init__(maestro)

        ttk.Label(self, text="Independencia lineal de vectores", font=("Segoe UI", 14, "bold")).pack(
            anchor="w", padx=10, pady=(0, 6)
        )

        controles = ttk.Frame(self)
        controles.pack(fill="x", padx=10, pady=10)

        self.m_dim = tk.IntVar(value=3)  # dimension m (filas)
        self.n_vec = tk.IntVar(value=3)  # cantidad de vectores n (columnas)

        ttk.Label(controles, text="Dimension (m):").pack(side="left")
        ttk.Spinbox(controles, from_=1, to=10, width=5, textvariable=self.m_dim).pack(side="left", padx=(4, 12))
        ttk.Label(controles, text="Numero de vectores (n):").pack(side="left")
        ttk.Spinbox(controles, from_=1, to=10, width=5, textvariable=self.n_vec).pack(side="left", padx=(4, 12))
        ttk.Button(controles, text="Actualizar", command=self._actualizar_dimensiones).pack(side="left", padx=(8, 0))
        ttk.Button(controles, text="Evaluar independencia", command=self._evaluar).pack(side="left", padx=(20, 0))

        self.boton_proceso = ttk.Button(controles, text="Ver proceso", command=self._mostrar_proceso, state="disabled")
        self.boton_proceso.pack(side="left", padx=(8, 0))

        # Entrada de datos: matriz m x n (columnas = vectores)
        marco_entrada = ttk.Frame(self)
        marco_entrada.pack(fill="x", padx=10, pady=10)
        self.entrada = EntradaMatriz(marco_entrada, filas=self.m_dim.get(), columnas=self.n_vec.get(),
                                     titulo="Matriz (columnas = vectores)")
        self.entrada.pack(side="left", padx=10)

        # Resultados
        self.marco_resultado = ttk.Frame(self)
        self.marco_resultado.pack(fill="both", expand=True, padx=10, pady=10)
        self._registro_final: Registro = []

    def _actualizar_dimensiones(self) -> None:
        self.entrada.establecer_dimensiones(int(self.m_dim.get()), int(self.n_vec.get()))

    def _evaluar(self) -> None:
        try:
            A = self.entrada.obtener_matriz()  # m x n
            m = len(A) if A else 0
            n = len(A[0]) if (A and A[0]) else 0

            # Construimos matriz aumentada [A | 0]
            # <--- 4. CAMBIO DE 0.0 a Fraction(0)
            A_aumentada = [fila + [Fraction(0)] for fila in A]

            registro: Registro = []
            rango, R, piv = rango_matriz(A, registro=registro)

            # Construimos también la RREF aumentada [R | 0]
            # <--- 4. CAMBIO DE 0.0 a Fraction(0)
            R_aumentada = [fila + [Fraction(0)] for fila in R]

            # Limpiamos la zona de resultados
            for w in self.marco_resultado.winfo_children():
                w.destroy()

            # Mostrar la matriz aumentada inicial
            mostrar_matriz(
                self.marco_resultado,
                A_aumentada,
                titulo="Matriz aumentada inicial [A | b], con b = 0"
            )

            # Mostrar la RREF aumentada
            mostrar_matriz(
                self.marco_resultado,
                R_aumentada,
                titulo="RREF aumentada [A | b], con b = 0"
            )

            # Evaluar independencia lineal
            independiente = (rango == n) and (m >= n)
            texto = (
                f"Dimensiones: m = {m}, n = {n}\n"
                f"Rango(A) = {rango}\n"
                f"Columnas pivote = "
                + (", ".join(str(c + 1) for c in piv) if piv else "[ninguna]")
                + "\n\n"
                + (
                    "… Los vectores son LINEALMENTE INDEPENDIENTES (rango = n)."
                    if independiente
                    else "¡  Los vectores son LINEALMENTE DEPENDIENTES (rango < n)."
                )
                + "\n Metodo aplicado: Eliminación de Gauss-Jordan sobre la matriz aumentada [A | 0]."
            )

            fila_texto = self.marco_resultado.grid_size()[1]
            ttk.Label(self.marco_resultado, text=texto, justify="left").grid(
                row=fila_texto,
                column=0,
                sticky="w",
                pady=(10, 0),
                padx=(0, 8),
            )

            self._registro_final = registro
            self.boton_proceso.config(state="normal")

        except Exception as error:
            messagebox.showerror("Error", str(error), parent=self)
            self.boton_proceso.config(state="disabled")


    def _mostrar_proceso(self) -> None:
        abrir_ventana_proceso(self, self._registro_final, titulo="Proceso: RREF / rango")

class VentanaTranspuesta(ttk.Frame):
    """Vista para calcular la transpuesta A^T."""

    def __init__(self, maestro):
        super().__init__(maestro)

        ttk.Label(self, text="Transpuesta de una matriz", font=("Segoe UI", 14, "bold")).pack(
            anchor="w", padx=10, pady=(0, 6)
        )

        self.filas = tk.IntVar(value=2)
        self.columnas = tk.IntVar(value=2)

        controles = ttk.Frame(self)
        controles.pack(fill="x", padx=10, pady=10)
        ttk.Label(controles, text="Filas A:").pack(side="left")
        ttk.Spinbox(controles, from_=1, to=10, width=5, textvariable=self.filas).pack(side="left", padx=(4, 12))
        ttk.Label(controles, text="Columnas A:").pack(side="left")
        ttk.Spinbox(controles, from_=1, to=10, width=5, textvariable=self.columnas).pack(side="left", padx=(4, 12))
        ttk.Button(controles, text="Actualizar", command=self._actualizar).pack(side="left", padx=(6, 0))
        ttk.Button(controles, text="Calcular A^T", command=self._calcular).pack(side="left", padx=(12, 0))
        self.btn_log = ttk.Button(controles, text="Ver proceso", command=self._ver_log, state="disabled")
        self.btn_log.pack(side="left", padx=(8, 0))

        cont = ttk.Frame(self); cont.pack(fill="x", padx=10, pady=10)
        self.entrada = EntradaMatriz(cont, 2, 2, titulo="Matriz A")
        self.entrada.pack(side="left", padx=10)

        self.salida = ttk.Frame(self); self.salida.pack(fill="both", expand=True, padx=10, pady=10)
        self._registro: Registro = []

    def _actualizar(self):
        self.entrada.establecer_dimensiones(int(self.filas.get()), int(self.columnas.get()))

    def _calcular(self):
        try:
            A = self.entrada.obtener_matriz()
            log: Registro = []
            AT = transpuesta(A, registro=log)
            mostrar_matriz(self.salida, AT, titulo="A^T (transpuesta)")
            self._registro = log
            self.btn_log.config(state="normal")
        except Exception as e:
            messagebox.showerror("Error", str(e), parent=self)
            self.btn_log.config(state="disabled")

    def _ver_log(self):
        abrir_ventana_proceso(self, self._registro, titulo="Proceso: transpuesta")

    # (GUI Determinante





class EntradaMatriz(ttk.Frame):
    """Componente de entrada de matriz mediante un arreglo de Entry."""

    def __init__(self, maestro, filas: int = 2, columnas: int = 2, titulo: str | None = None):
        super().__init__(maestro)
        self.filas = filas
        self.columnas = columnas
        self.titulo = titulo
        self._entradas: list[list[tk.Entry]] = []
        self._etiqueta_titulo: ttk.Label | None = None
        if titulo:
            self._etiqueta_titulo = ttk.Label(self, text=titulo, font=("Segoe UI", 10, "bold"))
            self._etiqueta_titulo.grid(row=0, column=0, columnspan=max(columnas, 1), pady=(0, 6))
        self._marco_celdas = ttk.Frame(self)
        self._marco_celdas.grid(row=1, column=0)
        self.construir_celdas()

    def establecer_dimensiones(self, filas: int, columnas: int) -> None:
        self.filas = max(0, int(filas))
        self.columnas = max(0, int(columnas))
        self.construir_celdas()

    def construir_celdas(self) -> None:
        for widget in self._marco_celdas.winfo_children():
            widget.destroy()
        self._entradas = []
        for indice_fila in range(self.filas):
            fila_entradas: list[tk.Entry] = []
            for indice_columna in range(self.columnas):
                entrada = ttk.Entry(self._marco_celdas, width=10, justify="center")
                entrada.grid(row=indice_fila, column=indice_columna, padx=2, pady=2)
                fila_entradas.append(entrada)
            self._entradas.append(fila_entradas)

    # <--- 3. FUNCION MODIFICADA (Solo el tipo de dato devuelto)
    def obtener_matriz(self) -> List[List[Fraction]]:
        matriz: List[List[Fraction]] = []
        for indice_fila in range(self.filas):
            fila_valores: List[Fraction] = []
            for indice_columna in range(self.columnas):
                try:
                    valor = convertir_numero(self._entradas[indice_fila][indice_columna].get())
                except ValueError as error:
                    raise ValueError(f"Valor invalido en ({indice_fila + 1},{indice_columna + 1})") from error
                fila_valores.append(valor)
            matriz.append(fila_valores)
        return matriz


def mostrar_matriz(contenedor: ttk.Frame, matriz: List[List[Fraction]], titulo: str | None = None) -> None:
    for widget in contenedor.winfo_children():
        widget.destroy()
    fila_actual = 0
    if titulo:
        columnas = max(len(matriz[0]) if matriz else 1, 1)
        ttk.Label(contenedor, text=titulo, font=("Segoe UI", 10, "bold")).grid(row=fila_actual, column=0, columnspan=columnas, pady=(0, 6))
        fila_actual += 1
    if not matriz:
        ttk.Label(contenedor, text="[vacio]").grid(row=fila_actual, column=0)
        return
    filas = len(matriz)
    columnas = len(matriz[0])
    anchos = []
    for columna in range(columnas):
        max_len = max(len(formatear_valor_ui(matriz[f][columna])) for f in range(filas))
        anchos.append(max(6, max_len))
    rejilla = ttk.Frame(contenedor)
    rejilla.grid(row=fila_actual, column=0)
    for indice_fila in range(filas):
        for indice_columna in range(columnas):
            valor_celda = matriz[indice_fila][indice_columna]
            texto = formatear_valor_ui(valor_celda)
            etiqueta = ttk.Label(
                rejilla,
                text=texto,
                width=anchos[indice_columna],
                anchor="center",
                relief="solid",
            )
            etiqueta.grid(row=indice_fila, column=indice_columna, padx=1, pady=1, ipadx=2, ipady=2, sticky="nsew")


def abrir_ventana_proceso(
    padre: tk.Misc,
    pasos: Sequence[RegistroEntrada],
    titulo: str = "Proceso de calculo",
) -> None:
    if not pasos:
        messagebox.showinfo("Proceso", "No hay proceso disponible todavia.", parent=padre)
        return

    ventana = tk.Toplevel(padre)
    ventana.title(titulo)
    ventana.geometry("960x640")

    contenedor = ttk.Frame(ventana, padding=12)
    contenedor.pack(fill="both", expand=True)

    canvas = tk.Canvas(contenedor, highlightthickness=0)
    barra_vertical = ttk.Scrollbar(contenedor, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=barra_vertical.set)

    barra_vertical.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    interior = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=interior, anchor="nw")

    def _actualizar_scroll(evento: tk.Event) -> None:
        canvas.configure(scrollregion=canvas.bbox("all"))

    interior.bind("<Configure>", _actualizar_scroll)

    first = True
    for paso in pasos:
        if isinstance(paso, tuple) and len(paso) == 3 and paso[0] == "matrix":
            _, titulo_matriz, matriz = paso
            if not first:
                ttk.Separator(interior, orient="horizontal").pack(fill="x", pady=6)
            seccion = ttk.Frame(interior, padding=(0, 6))
            seccion.pack(fill="x", anchor="w")
            if titulo_matriz:
                ttk.Label(seccion, text=titulo_matriz, font=("Segoe UI", 11, "bold")).pack(anchor="w")
            marco_matriz = ttk.Frame(seccion)
            marco_matriz.pack(anchor="w", pady=(4, 0))
            mostrar_matriz(marco_matriz, matriz, titulo=None)
        else:
            texto = str(paso).strip("\n")
            if not texto:
                continue
            if not first:
                ttk.Separator(interior, orient="horizontal").pack(fill="x", pady=6)
            ttk.Label(interior, text=texto, justify="left", wraplength=860).pack(
                fill="x", anchor="w", pady=(0, 4)
            )
        first = False
    
   

def abrir_ventana_tabla_biseccion(maestro, registro: RegistroBiseccion, titulo: str):
    """Abre una ventana Toplevel para mostrar el registro de Bisección como tabla (Treeview)."""
    if not registro:
        messagebox.showinfo("Proceso", "No hay registro de iteraciones para mostrar.", parent=maestro)
        return

    ventana = tk.Toplevel(maestro)
    ventana.title(titulo)
    ventana.geometry("900x600")
    ventana.minsize(600, 300)

    # --- Treeview para la tabla ---
    # Definimos las columnas que mostraremos
    columnas = ("iter", "a", "f(a)", "b", "f(b)", "c", "f(c)", "error")
    
    # Frame para el Treeview y Scrollbars
    frame_tabla = ttk.Frame(ventana)
    frame_tabla.pack(fill="both", expand=True, padx=10, pady=10)

    arbol = ttk.Treeview(frame_tabla, columns=columnas, show="headings")
    
    # Definir encabezados
    arbol.heading("iter", text="Iter")
    arbol.heading("a",    text="a")
    arbol.heading("f(a)", text="f(a)")
    arbol.heading("b",    text="b")
    arbol.heading("f(b)", text="f(b)")
    arbol.heading("c",    text="c (raíz aprox.)")
    arbol.heading("f(c)", text="f(c)")

    arbol.heading("error", text="Error (b-a)/2")
    
    # Definir anchos y alineación de columna
    arbol.column("iter", width=40, anchor="center", stretch=False)
    for col in ("a","f(a)","b","f(b)","c","f(c)","error"):
        arbol.column(col, width=140, anchor="e")
    arbol.column("a", width=140, anchor="e")
    arbol.column("b", width=140, anchor="e")
    arbol.column("c", width=140, anchor="e")
    arbol.column("f(c)", width=140, anchor="e")
    arbol.column("error", width=140, anchor="e")

    # --- Scrollbars ---
    v_scroll = ttk.Scrollbar(frame_tabla, orient="vertical", command=arbol.yview)
    h_scroll = ttk.Scrollbar(frame_tabla, orient="horizontal", command=arbol.xview)
    arbol.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

    v_scroll.pack(side="right", fill="y")
    h_scroll.pack(side="bottom", fill="x")
    arbol.pack(side="left", fill="both", expand=True)

    # --- Insertar datos ---
    # Usamos formato de punto flotante para la tabla, es más legible
    precision_g = 8 
    
    for fila_datos in registro:
        valores_formateados = (
            fila_datos["iter"],
                f'{float(fila_datos["a"]):.{precision_g}g}',
                f'{float(fila_datos["f(a)"]):.{precision_g}g}',
                f'{float(fila_datos["b"]):.{precision_g}g}',
                f'{float(fila_datos["f(b)"]):.{precision_g}g}',
                f'{float(fila_datos["c"]):.{precision_g}g}',
                f'{float(fila_datos["f(c)"]):.{precision_g}g}',
                f'{float(fila_datos["error"]):.{precision_g}g}',
        )
        arbol.insert("", "end", values=valores_formateados)
        
    ventana.transient(maestro)
    ventana.grab_set()
    maestro.wait_window(ventana)




class VentanaBiseccion(ttk.Frame):
    """Vista para el Metodo de Biseccion, con gráfico y panel de entrada."""

    titulo = "Metodo de Biseccion"
    descripcion = "Encuentra una raiz de f(x) = 0 en el intervalo [a, b]."
    metodo_solver = metodo_biseccion


    def __init__(self, maestro):
        super().__init__(maestro, padding=12)
        self._titulo = getattr(self, "titulo", "Metodo de Biseccion")
        self._descripcion = getattr(self, "descripcion", "Encuentra una raiz de f(x) = 0 en el intervalo [a, b].")
        
        # --- Layout Principal (Panel dividido) ---
        # Panel que divide la ventana en dos (controles a la izq, gráfico a la der)
        self.panel_principal = ttk.PanedWindow(self, orient="horizontal")
        self.panel_principal.pack(fill="both", expand=True)

        # --- 1. Panel Izquierdo (Controles) ---
        self.frame_controles = ttk.Frame(self.panel_principal, padding=12)
        self.panel_principal.add(self.frame_controles, weight=1) # 'weight=1' le da prioridad

        ttk.Label(self.frame_controles, text=self._titulo, font=("Segoe UI", 14, "bold")).pack(
            anchor="w", pady=(0, 8)
        )
        ttk.Label(self.frame_controles, text=self._descripcion).pack(
            anchor="w", pady=(0, 12)
        )

        # --- Controles de Entrada ---
        controles = ttk.Frame(self.frame_controles)
        controles.pack(fill="x", pady=5)
        
        # Fila 1: f(x)
        ttk.Label(controles, text="f(x) =").grid(row=0, column=0, padx=(0, 5), pady=4, sticky="e")
        self.fx_var = tk.StringVar(value="x^3 - x - 2")
        # Guardamos la 'Entry' para poder insertar texto en ella
        self.fx_entry = ttk.Entry(controles, textvariable=self.fx_var, width=30)
        self.fx_entry.grid(row=0, column=1, pady=4, sticky="we")

        # --- Auto-grafica al escribir f(x) (debounce) ---
        self._after_id = None
        self.fx_var.trace_add("write", lambda *args: self._programar_redibujo())
        self.fx_entry.bind("<KeyRelease>", lambda e: self._programar_redibujo())
        
        # --- (RETO 2) Panel de Botones para f(x) ---
        self.crear_panel_botones_fx(controles)

        # Fila 2: a, b
        ttk.Label(controles, text="Intervalo [a, b]:").grid(row=3, column=0, padx=(0, 5), pady=4, sticky="e")
        self.a_var = tk.StringVar(value="1")
        self.b_var = tk.StringVar(value="2")

        # Traza cambios en a y b para refrescar marcadores en el gráfico
        self.a_var.trace_add("write", lambda *args: self._actualizar_marcadores_intervalo())
        self.b_var.trace_add("write", lambda *args: self._actualizar_marcadores_intervalo())

        frame_ab = ttk.Frame(controles)
        frame_ab.grid(row=3, column=1, pady=4, sticky="w")
        ttk.Label(frame_ab, text="a =").pack(side="left", padx=(0, 4))
        ttk.Entry(frame_ab, textvariable=self.a_var, width=12).pack(side="left")
        ttk.Label(frame_ab, text="  b =").pack(side="left", padx=(8, 4))
        ttk.Entry(frame_ab, textvariable=self.b_var, width=12).pack(side="left")
        
        # Fila 3: Tolerancia
        ttk.Label(controles, text="Tolerancia:").grid(row=4, column=0, padx=(0, 5), pady=4, sticky="e")
        self.tol_var = tk.StringVar(value="0.0001")
        ttk.Entry(controles, textvariable=self.tol_var, width=12).grid(row=4, column=1, pady=4, sticky="w")
        
        # Fila 4: Max Iter
        ttk.Label(controles, text="Max. Iteraciones:").grid(row=5, column=0, padx=(0, 5), pady=4, sticky="e")
        self.max_iter_var = tk.IntVar(value=100)
        ttk.Spinbox(controles, from_=10, to=1000, width=10, textvariable=self.max_iter_var).grid(row=5, column=1, pady=4, sticky="w")

        controles.columnconfigure(1, weight=1)

        # --- Botones de Acción ---
        botones = ttk.Frame(self.frame_controles)
        botones.pack(fill="x", pady=(8, 0))
        ttk.Button(botones, text="Calcular Raiz", command=self._calcular).pack(side="left")
        ttk.Button(botones, text="Limpiar", command=self._limpiar_resultado).pack(side="left", padx=(8, 0))
        self.boton_proceso = ttk.Button(botones, text="Ver Proceso", command=self._mostrar_proceso, state="disabled")
        self.boton_proceso.pack(side="left", padx=(8, 0))

        # --- Resultados ---
        self.marco_resultado = ttk.Frame(self.frame_controles)
        self.marco_resultado.pack(fill="both", expand=True, pady=(12, 0))

        # --- 2. Panel Derecho (Gráfico) ---
        self.frame_grafico_main = ttk.Frame(self.panel_principal, padding=(12, 0, 0, 0))
        self.panel_principal.add(self.frame_grafico_main, weight=2) # 'weight=2' le da más espacio

        # --- Gráfico de Matplotlib ---
        self.figura = Figure(figsize=(5, 4), dpi=100, facecolor="#f0f0f0")
        self.ax = self.figura.add_subplot(1, 1, 1)
        self.ax.set_facecolor("#ffffff")
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.ax.format_coord = lambda x, y: ""
        
        self.canvas = FigureCanvasTkAgg(self.figura, master=self.frame_grafico_main)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame_grafico_main, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")
        mensaje = getattr(self.toolbar, "_message_label", None)
        if mensaje is not None:
            mensaje.pack_forget()

        # Debounce para redibujar al cambiar el viewport
        self._view_after_id = None
        self.ax.callbacks.connect('xlim_changed', lambda ax: self._on_view_change())
        self.ax.callbacks.connect('ylim_changed', lambda ax: self._on_view_change())

        # --- Estado para selección con clic en la gráfica ---
        self._linea_a = None
        self._linea_b = None
        self._sombreado = None
        self._f_numpy = None
        self._rango_actual = (-10.0, 10.0)
        self._curve_line, = self.ax.plot([], [], linewidth=2, label="f(x)")

        # Dibujo inicial (gráfica en vivo)
        self._limpiar_grafico()
        self._auto_graficar()

        # Fija la posición inicial del divisor (350px desde la izquierda)
        self.panel_principal.sashpos(0, 350)

        # --- Estado interno ---
        self._registro_final: RegistroBiseccion = []
        self._resultado_final: Optional[Fraction] = None
        

    def crear_panel_botones_fx(self, maestro):
        """(RETO 2) Crea el panel de botones para f(x)"""
        panel = ttk.Frame(maestro)
        panel.grid(row=1, column=0, columnspan=2, pady=5, sticky="we")
        
        # Fila 1 de botones
        botones_fx = [
            ("x^2", "x^2"), ("x^3", "x^3"), ("( )", "()"), ("√", "sqrt()"), ("|x|", "abs()"),
        ]
        for i, (texto_btn, texto_insert) in enumerate(botones_fx):
            cmd = lambda t=texto_insert: self._insertar_texto(t)
            ttk.Button(panel, text=texto_btn, command=cmd, width=4).grid(row=0, column=i, padx=2, pady=2)
            
        # Fila 2 de botones
        botones_fx_2 = [
            ("sin", "sin()"), ("cos", "cos()"), ("tan", "tan()"), ("log", "log()"), ("exp", "exp()"),
        ]
        for i, (texto_btn, texto_insert) in enumerate(botones_fx_2):
            cmd = lambda t=texto_insert: self._insertar_texto(t)
            ttk.Button(panel, text=texto_btn, command=cmd, width=4).grid(row=1, column=i, padx=2, pady=2)
            
        # Mueve el cursor dentro de paréntesis/sqrt
        self.fx_entry.bind("<ButtonRelease-1>", self._mover_cursor_helper)
        self.fx_entry.bind("<KeyRelease>", self._mover_cursor_helper)

    def _insertar_texto(self, texto: str):
        """(RETO 2) Inserta texto en el Entry de f(x) en la posición del cursor."""
        try:
            self.fx_entry.focus()
            posicion_cursor = self.fx_entry.index(tk.INSERT)
            self.fx_entry.insert(posicion_cursor, texto)
            if texto.endswith("()"):
                self.fx_entry.icursor(posicion_cursor + len(texto) - 1)
        except Exception as e:
            print(f"Error al insertar texto: {e}")

    def _mover_cursor_helper(self, event=None):
        """(RETO 2) Mueve el cursor si se hace clic justo después de '()'"""
        try:
            pos = self.fx_entry.index(tk.INSERT)
            if pos > 0:
                char_anterior = self.fx_entry.get()[pos-1]
                if char_anterior == ')':
                    if self.fx_entry.get()[pos:pos+1] == ')':
                        self.fx_entry.icursor(pos)
        except Exception:
            pass

    def _limpiar_resultado(self):
        """Limpia la zona de resultados y deshabilita el botón de proceso."""
        for w in self.marco_resultado.winfo_children():
            w.destroy()
        self.boton_proceso.config(state="disabled")
        self._registro_final = []
        self._resultado_final = None
        self._limpiar_grafico()
        # No borramos a/b a propósito; solo limpiamos el dibujo

    def _limpiar_grafico(self):
        """Limpia la gráfica pero deja lista la línea reutilizable."""
        self.ax.cla()
        self.ax.set_facecolor("#ffffff")
        self.ax.grid(True, linestyle="--", alpha=0.6)
        self.ax.axhline(0, color="black", linewidth=0.7)
        self.ax.axvline(0, color="black", linewidth=0.7)
        self.ax.set_title("GrA?fico de f(x)")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("f(x)")
        self.ax.format_coord = lambda x, y: ""
        self._linea_a = self._linea_b = self._sombreado = None
        etiqueta = getattr(self, "_curve_line", None)
        etiqueta = etiqueta.get_label() if etiqueta else "f(x)"
        self._curve_line, = self.ax.plot([], [], linewidth=2, label=etiqueta)
        self.ax.legend(loc="best")
        self.canvas.draw_idle()

    # =========================
    #   NUEVO: Gráfica en vivo
    # =========================
    def _programar_redibujo(self):
        """Programa un redibujo con *debounce* para no congelar la UI."""
        if getattr(self, "_after_id", None) is not None:
            try:
                aid = self._after_id
                if aid is not None:
                    self.after_cancel(aid)
            except Exception:
                pass
        self._after_id = self.after(250, self._auto_graficar)

    def _auto_graficar(self):
        expr = (self.fx_var.get() or "").strip()
        if not expr:
            self._f_numpy = None
            self._curve_line.set_data([], [])
            self.ax.set_title("GrA?fico de f(x)")
            self.canvas.draw_idle()
            return
        try:
            self._f_numpy = _crear_evaluador_numpy(expr)
            x_min, x_max = self._sugerir_rango_x(self._f_numpy)
            self.ax.set_xlim(x_min, x_max)
            self._rango_actual = (min(x_min, x_max), max(x_min, x_max))
            self._zoom_home = (min(x_min, x_max), max(x_min, x_max))
            self._curve_line.set_label(f"f(x) = {expr}")
            self.ax.legend(loc="best")
            self._redraw_visible_curve()
            self._actualizar_marcadores_intervalo()
        except Exception as e:
            self.ax.set_title(f"Esperando una expresiA3n vA?lida??? ({e})")
            self.canvas.draw_idle()

    def _on_view_change(self):
        """Debounce: espera un instante y redibuja la curva en el rango visible."""
        if getattr(self, "_view_after_id", None):
            try:
                self.after_cancel(self._view_after_id)
            except Exception:
                pass
        self._view_after_id = self.after(90, self._redraw_visible_curve)

    def _ajustar_zoom(self, factor: float):
        """Aplica zoom conservando el centro visible actual."""
        try:
            x0, x1 = self.ax.get_xlim()
        except Exception:
            return
        if not np.isfinite([x0, x1]).all() or x0 == x1:
            return
        centro = 0.5 * (x0 + x1)
        mitad = max(1e-9, 0.5 * (x1 - x0) * factor)
        self.ax.set_xlim(centro - mitad, centro + mitad)
        self._redraw_visible_curve()

    def _restaurar_zoom(self):
        """Vuelve al rango automático sugerido para la función actual."""
        home = getattr(self, "_zoom_home", None)
        if not home:
            return
        self.ax.set_xlim(*home)
        self._redraw_visible_curve()

    def _redraw_visible_curve(self):
        f = getattr(self, "_f_numpy", None)
        if f is None:
            return

        home = getattr(self, "_zoom_home", None)
        if home:
            lo, hi = home
        else:
            x0, x1 = self.ax.get_xlim()
            if not np.all(np.isfinite([x0, x1])) or x0 == x1:
                return
            lo, hi = (x0, x1) if x0 < x1 else (x1, x0)

        x = np.linspace(lo, hi, 1200)
        y = np.asarray(f(x), dtype=float)
        finite = np.isfinite(y)
        if not finite.any():
            self._curve_line.set_data([], [])
            self.canvas.draw_idle()
            return
        y = np.where(finite, y, np.nan)
        self._curve_line.set_data(x, y)
        self.ax.relim()
        self.ax.autoscale_view(scalex=False, scaley=True)
        self._rango_actual = (lo, hi)
        self.canvas.draw_idle()

    def _sugerir_rango_x(self, f_numpy):
        """Busca un rango donde la función sea visible y tenga variación."""
        rangos = [(-1,1), (-2,2), (-5,5), (-10,10), (-25,25), (-50,50)]
        for lo, hi in rangos:
            x = np.linspace(lo, hi, 600)
            y = f_numpy(x)
            finite = np.isfinite(y)
            if finite.sum() > 50:
                ys = y[finite]
                if ys.size and not np.allclose(ys, ys[0]):
                    return float(lo), float(hi)
        return -10.0, 10.0

    # =======================================
    #   NUEVO: Selección de [a,b] con clics
    # =======================================
    def _actualizar_marcadores_intervalo(self):
        """Dibuja líneas y sombreado de [a,b]; verde si hay cambio de signo."""
        # Borrar marcadores previos
        for line in (getattr(self, "_linea_a", None), getattr(self, "_linea_b", None)):
            if line is not None:
                try:
                    line.remove()
                except Exception:
                    pass
        # Safely remove shaded span if it exists and provides a remove() method
        sombra = getattr(self, "_sombreado", None)
        if sombra is not None:
            try:
                remover = getattr(sombra, "remove", None)
                if callable(remover):
                    remover()
            except Exception:
                pass
        self._linea_a = self._linea_b = self._sombreado = None

        if getattr(self, "_f_numpy", None) is None:
            return
        # Obtener a,b
        try:
            a = float((self.a_var.get() or "").replace(",", "."))
            b = float((self.b_var.get() or "").replace(",", "."))
        except Exception:
            return
        if not (np.isfinite(a) and np.isfinite(b)):
            return

        lo, hi = (a, b) if a <= b else (b, a)
        self._linea_a = self.ax.axvline(lo, linestyle="--", linewidth=1.2)
        self._linea_b = self.ax.axvline(hi, linestyle="--", linewidth=1.2)

        # Color del sombreado según cambio de signo
        if self._f_numpy is not None:
            fa = self._f_numpy(np.array([lo], dtype=float))
            fb = self._f_numpy(np.array([hi], dtype=float))
            fa = float(fa[0]) if isinstance(fa, np.ndarray) else float(fa)
            fb = float(fb[0]) if isinstance(fb, np.ndarray) else float(fb)
            cambia = np.isfinite(fa) and np.isfinite(fb) and (np.sign(fa) * np.sign(fb) <= 0)
        else:
            cambia = False
        color = (0.2, 0.7, 0.2, 0.15) if cambia else (0.8, 0.2, 0.2, 0.12)
        self._sombreado = self.ax.axvspan(lo, hi, color=color)

        # Mantener el rango X estable si se calculó
        if getattr(self, "_rango_actual", None) is not None:
            self.ax.set_xlim(*self._rango_actual)

    # =========================
    #   cálculo y visualización
    # =========================
    def _calcular(self) -> None:
        """Manejador del boton 'Calcular Raíz'."""
        self._limpiar_resultado()
        
        try:
            expresion = self.fx_var.get()
            a_frac = convertir_numero(self.a_var.get())
            b_frac = convertir_numero(self.b_var.get())
            tol_frac = convertir_numero(self.tol_var.get())
            max_iter = int(self.max_iter_var.get())
            
            solver = getattr(type(self), "metodo_solver", metodo_biseccion)
            raiz, registro = solver(expresion, a_frac, b_frac, tol_frac, max_iter)
            self._registro_final = registro
            self._resultado_final = raiz
            
            # Mostrar resultado (texto)
            texto_resultado = (
                f"Raíz aproximada encontrada:\n"
                f"c ≈ {formatear_valor_ui(raiz)}\n\n"
                f"(Valor numérico: {float(raiz):.12g})\n\n"
                f"Iteraciones: {len(registro)}"
            )
            ttk.Label(self.marco_resultado, text=texto_resultado, font=("Segoe UI", 11), justify="left").pack(anchor="w")
            self.boton_proceso.config(state="normal")
            
            # Dibujar el gráfico focalizado en [a,b] y la raíz
            self._dibujar_grafica(expresion, a_frac, b_frac, raiz)

        except Exception as error:
            messagebox.showerror("Error de Cálculo", str(error), parent=self)
            self.boton_proceso.config(state="disabled")

    def _dibujar_grafica(self, expresion, a, b, raiz):
        """Dibuja la función y la raíz en el canvas de Matplotlib."""
        self._limpiar_grafico()
        try:
            f_numpy = _crear_evaluador_numpy(expresion)
            
            # Convertir a float para graficar
            a_f, b_f, r_f = float(a), float(b), float(raiz)
            
            # Crear un rango de 1000 puntos para la gráfica (20% de aire a cada lado)
            ancho = b_f - a_f
            x_min = a_f - (ancho * 0.2)
            x_max = b_f + (ancho * 0.2)
            x_vals = np.linspace(x_min, x_max, 1000)
            y_vals = f_numpy(x_vals)

            # Ejes
            self.ax.axhline(0, color='black', linewidth=0.7)
            self.ax.axvline(0, color='black', linewidth=0.7)
            self.ax.grid(True, linestyle='--', alpha=0.6)

            # Curva
            self.ax.plot(x_vals, y_vals, label=f"$f(x) = {expresion.replace('**', '^')}$")

            # Intervalo y raíz
            self.ax.axvline(a_f, color='tab:orange', linestyle='--', linewidth=1.2, label='a')
            self.ax.axvline(b_f, color='tab:orange', linestyle='--', linewidth=1.2, label='b')
            self.ax.axvspan(min(a_f, b_f), max(a_f, b_f), color=(0.2, 0.5, 0.9, 0.10))
            self.ax.plot([r_f], [0], marker='o', markersize=6, color='tab:red', label='raíz aprox.')

            # Ajustar límites de Y para que se vea bien
            y_visibles = y_vals[~np.isnan(y_vals)]
            if y_visibles.size > 0:
                y_min, y_max = np.min(y_visibles), np.max(y_visibles)
                y_rango = y_max - y_min
                if y_min > 0: y_min = -0.1 * y_rango
                if y_max < 0: y_max = 0.1 * y_rango
                self.ax.set_ylim(y_min - 0.1 * y_rango, y_max + 0.1 * y_rango)

            self.ax.set_title("Gráfico de $f(x)$")
            self.ax.legend()
            self.canvas.draw()
            
        except Exception as e:
            self.ax.set_title(f"Error al graficar: {e}")
            self.canvas.draw()

    def _mostrar_proceso(self) -> None:
        """Manejador del botón 'Ver Proceso'."""
        if not self._registro_final:
            messagebox.showwarning("Proceso", "No hay datos de proceso para mostrar.", parent=self)
            return
        titulo_proceso = f"Proceso: {self._titulo} f(x)={self.fx_var.get()}"
        abrir_ventana_tabla_biseccion(self, self._registro_final, titulo_proceso)



class VentanaReglaFalsa(VentanaBiseccion):
    """Ventana especializada para el Metodo de Regla Falsa."""

    titulo = "Metodo de Regla Falsa"
    descripcion = "Encuentra una raiz de f(x) = 0 usando la Regla Falsa en [a, b]."
    metodo_solver = metodo_regla_falsa

class VentanaSecante(VentanaBiseccion):
    """Ventana para el Método de la Secante (versión falsa posición adaptada)."""

    titulo = "Método de la Secante"
    descripcion = "Encuentra una raíz aproximada usando la secante con (x0, x1)."
    metodo_solver = secante_falsaPosicion

    def __init__(self, maestro):
        super().__init__(maestro)

        # Cambiar etiquetas del intervalo por x0 y x1
        for w in self.frame_controles.winfo_children():
            w.destroy()

        ttk.Label(self.frame_controles, text=self.titulo, font=("Segoe UI", 14, "bold")).pack(
            anchor="w", pady=(0, 8)
        )
        ttk.Label(self.frame_controles, text=self.descripcion).pack(
            anchor="w", pady=(0, 12)
        )

        controles = ttk.Frame(self.frame_controles)
        controles.pack(fill="x", pady=5)

        # Función
        ttk.Label(controles, text="f(x) =").grid(row=0, column=0, padx=(0,5), pady=4, sticky="e")
        self.fx_var = tk.StringVar(value="x^3 - x - 2")
        self.fx_entry = ttk.Entry(controles, textvariable=self.fx_var, width=30)
        self.fx_entry.grid(row=0, column=1, pady=4, sticky="we")

        self._after_id = None
        self.fx_var.trace_add("write", lambda *args: self._programar_redibujo())
        self.fx_entry.bind("<KeyRelease>", lambda e: self._programar_redibujo())

        self.crear_panel_botones_fx(controles)

        # x0 y x1
        ttk.Label(controles, text="Puntos iniciales (x0, x1):").grid(
            row=3, column=0, padx=(0,5), pady=4, sticky="e"
        )
        self.x0_var = tk.StringVar(value="1")
        self.x1_var = tk.StringVar(value="2")

        fila = ttk.Frame(controles)
        fila.grid(row=3, column=1, pady=4, sticky="w")
        ttk.Label(fila, text="x0 =").pack(side="left", padx=(0, 4))
        ttk.Entry(fila, textvariable=self.x0_var, width=12).pack(side="left")
        ttk.Label(fila, text="  x1 =").pack(side="left", padx=(8, 4))
        ttk.Entry(fila, textvariable=self.x1_var, width=12).pack(side="left")

        # Tolerancia
        ttk.Label(controles, text="Tolerancia:").grid(row=4, column=0, sticky="e")
        self.tol_var = tk.StringVar(value="0.0001")
        ttk.Entry(controles, textvariable=self.tol_var, width=12).grid(row=4, column=1, sticky="w")

        # Iteraciones
        ttk.Label(controles, text="Max. Iteraciones:").grid(row=5, column=0, sticky="e")
        self.max_iter_var = tk.IntVar(value=100)
        ttk.Spinbox(controles, from_=10, to=1000, width=10, textvariable=self.max_iter_var).grid(
            row=5, column=1, sticky="w"
        )

        controles.columnconfigure(1, weight=1)

        # Botones
        botones = ttk.Frame(self.frame_controles)
        botones.pack(fill="x", pady=(8, 0))
        ttk.Button(botones, text="Calcular Raíz", command=self._calcular).pack(side="left")
        ttk.Button(botones, text="Limpiar", command=self._limpiar_resultado).pack(side="left", padx=(8, 0))
        self.boton_proceso = ttk.Button(botones, text="Ver Proceso", state="disabled", command=self._mostrar_proceso)
        self.boton_proceso.pack(side="left", padx=(8, 0))

        self.marco_resultado = ttk.Frame(self.frame_controles)
        self.marco_resultado.pack(fill="both", expand=True, pady=(12, 0))

    def _calcular(self):
        self._limpiar_resultado()
        try:
            expresion = self.fx_var.get()
            x0 = convertir_numero(self.x0_var.get())
            x1 = convertir_numero(self.x1_var.get())
            tol = convertir_numero(self.tol_var.get())
            max_iter = int(self.max_iter_var.get())

            # Evitar que la función de solver se convierta en un método ligado a la instancia
            solver = getattr(type(self), "metodo_solver", secante_falsaPosicion)
            # secante_falsaPosicion espera floats, convertir Fraction -> float aquí
            raiz, registro = solver(expresion, float(x0), float(x1), float(tol), max_iter)
            self._registro_final = registro
            # Asegurar que _resultado_final tenga tipo Fraction | None como se declara en la clase base
            try:
                # Si ya es Fraction lo dejamos, si es float/otro lo convertimos
                self._resultado_final = raiz if isinstance(raiz, Fraction) else Fraction(raiz)
            except Exception:
                # Como último recurso, intentar convertir a partir de la representación en cadena
                self._resultado_final = Fraction(str(raiz))

            texto = (
                f"Raíz aproximada:\n"
                f"x ≈ {formatear_valor_ui(raiz)}\n\n"
                f"(Valor numérico: {float(raiz):.12g})\n"
                f"Iteraciones: {len(registro)}"
            )
            ttk.Label(self.marco_resultado, text=texto, font=("Segoe UI", 11)).pack(anchor="w")
            self.boton_proceso.config(state="normal")
            self._dibujar_grafica(expresion, float(x0), float(x1), float(raiz))

        except Exception as e:
            messagebox.showerror("Error", str(e), parent=self)
            self.boton_proceso.config(state="disabled")

    def _mostrar_proceso(self): 

        abrir_ventana_proceso_secante(self, self._registro_final, titulo="Proceso – Método de la Secante")


def abrir_ventana_proceso_secante(
    parent: tk.Misc, 
    registro: RegistroBiseccion, # El tipo 'RegistroBiseccion' ya está importado
    titulo: str = "Proceso: Metodo de la Secante"
) -> None:
    """Ventana con la tabla de iteraciones específica para la Secante."""
    from tkinter import ttk, messagebox
    
    if not registro:
        messagebox.showinfo("Proceso", "No hay proceso disponible todavía.", parent=parent)
        return
        
    win = tk.Toplevel(parent)
    win.title(titulo)
    win.geometry("900x450")
    
    # 1. Definición de columnas y Treeview
    # Las columnas son específicas de la Secante
    cols = ("iter", "xi_1", "f_xi_1", "xi", "f_xi", "xi_mas_1", "error")
    tabla = ttk.Treeview(win, columns=cols, show="headings", height=20) 
    
    encabezados = {
        "iter": "Iter", 
        "xi_1": "xi-1", 
        "f_xi_1": "f(xi-1)", 
        "xi": "xi", 
        "f_xi": "f(xi)", 
        "xi_mas_1": "xi+1", 
        "error": "Error"
    }

    for col in cols:
        tabla.heading(col, text=encabezados.get(col, col))
        tabla.column(col, width=120, anchor="center")
    
    tabla.column("iter", width=40, anchor="center")
    tabla.column("error", width=120, anchor="center") 
    
    # 2. Rellenar la tabla con los datos del registro
    precision_g = 8 
    
    for fila_datos in registro:
        try:
            # Los nombres de las claves son los que devuelve secante_falsaPosicion
            valores_formateados = (
                fila_datos.get("iter", ""),
                f'{float(fila_datos.get("xi-1", 0)):.{precision_g}g}',
                f'{float(fila_datos.get("f(xi-1)", 0)):.{precision_g}g}',
                f'{float(fila_datos.get("xi", 0)):.{precision_g}g}',
                f'{float(fila_datos.get("f(xi)", 0)):.{precision_g}g}',
                f'{float(fila_datos.get("xi+1", 0)):.{precision_g}g}',
                f'{float(fila_datos.get("error", 0)):.{precision_g}g}'
            )
        except (ValueError, TypeError):
             # Manejo de casos donde el 'error' es un string de aviso
            valores_formateados = (
                str(fila_datos.get("iter", "")),
                str(fila_datos.get("xi-1", "")),
                str(fila_datos.get("f(xi-1)", "")),
                str(fila_datos.get("xi", "")),
                str(fila_datos.get("f(xi)", "")),
                str(fila_datos.get("xi+1", "")),
                str(fila_datos.get("error", ""))
            )
        
        tabla.insert("", "end", values=valores_formateados)

    # 3. Mostrar y configurar como ventana modal
    tabla.pack(fill="both", expand=True, padx=10, pady=10)
    win.transient(parent)
    win.grab_set()
    parent.wait_window(win)




class PanelMatriz(ttk.Frame):
    """Panel con controles de dimensiones y un componente EntradaMatriz."""

    def __init__(self, maestro, indice: int, filas_iniciales: int = 2, columnas_iniciales: int = 2, prefijo_titulo: str = "Matriz"):
        super().__init__(maestro, relief="groove", padding=6)
        self.indice = indice
        self.filas = tk.IntVar(value=filas_iniciales)
        self.columnas = tk.IntVar(value=columnas_iniciales)

        cabecera = ttk.Frame(self)
        cabecera.pack(fill="x")
        ttk.Label(cabecera, text=f"{prefijo_titulo} {indice}", font=("Segoe UI", 10, "bold")).pack(side="left")
        controles_dim = ttk.Frame(cabecera)
        controles_dim.pack(side="right")
        ttk.Label(controles_dim, text="Filas:").pack(side="left")
        self.control_filas = ttk.Spinbox(controles_dim, from_=1, to=10, width=5, textvariable=self.filas)
        self.control_filas.pack(side="left", padx=(4, 8))
        ttk.Label(controles_dim, text="Columnas:").pack(side="left")
        self.control_columnas = ttk.Spinbox(controles_dim, from_=1, to=10, width=5, textvariable=self.columnas)
        self.control_columnas.pack(side="left", padx=(4, 0))

        self.entrada = EntradaMatriz(self, filas_iniciales, columnas_iniciales)
        self.entrada.pack(fill="x", pady=(6, 0))

    def aplicar_dimensiones(self) -> None:
        self.entrada.establecer_dimensiones(int(self.filas.get()), int(self.columnas.get()))

    def obtener_matriz(self) -> List[List[Fraction]]:
        return self.entrada.obtener_matriz()

class VentanaEjercicioTranspuestas(ttk.Frame):
    """Vista para (Ax)^T, x^T A^T, x x^T, x^T x y opcional y^T x."""

    def __init__(self, maestro):
        super().__init__(maestro)

        ttk.Label(self, text="Ejercicio: propiedades de la transpuesta", font=("Segoe UI", 14, "bold")).pack(
            anchor="w", padx=10, pady=(0, 6)
        )

        # Dimensiones: A(mxn), x(nx1) y y(nx1 opcional)
        self.m = tk.IntVar(value=2)
        self.n = tk.IntVar(value=2)

        barra = ttk.Frame(self)
        barra.pack(fill="x", padx=10, pady=10)
        ttk.Label(barra, text="Filas A (m):").pack(side="left")
        ttk.Spinbox(barra, from_=1, to=8, width=4, textvariable=self.m).pack(side="left", padx=(4, 10))
        ttk.Label(barra, text="Columnas A = tam(x) (n):").pack(side="left")
        ttk.Spinbox(barra, from_=1, to=8, width=4, textvariable=self.n).pack(side="left", padx=(4, 10))
        ttk.Button(barra, text="Actualizar", command=self._actualizar).pack(side="left", padx=(6, 0))
        ttk.Button(barra, text="Calcular", command=self._calcular).pack(side="left", padx=(12, 0))
        self.btn_log = ttk.Button(barra, text="Ver proceso", state="disabled", command=self._ver_log)
        self.btn_log.pack(side="left", padx=(8, 0))

        # Entradas
        paneles = ttk.Frame(self); paneles.pack(fill="x", padx=10, pady=10)
        self.ent_A = EntradaMatriz(paneles, 2, 2, titulo="Matriz A (m x n)"); self.ent_A.pack(side="left", padx=10)
        self.ent_x = EntradaMatriz(paneles, 2, 1, titulo="Vector x (n x 1)"); self.ent_x.pack(side="left", padx=10)
        self.ent_y = EntradaMatriz(paneles, 2, 1, titulo="Vector y (opcional, n x 1)"); self.ent_y.pack(side="left", padx=10)

        # Relleno de ejemplo del enunciado:
        # A = [[1, -3], [-2, 4]] y x = [[5],[3]]
        try:
            self.ent_A.establecer_dimensiones(2, 2); self.ent_x.establecer_dimensiones(2, 1); self.ent_y.establecer_dimensiones(2, 1)
            for (r, c, v) in [(0, 0, 1), (0, 1, -3), (1, 0, -2), (1, 1, 4)]:
                self.ent_A._entradas[r][c].delete(0, "end")
                self.ent_A._entradas[r][c].insert(0, str(v))
            for (r, v) in [(0, 5), (1, 3)]:
                self.ent_x._entradas[r][0].delete(0, "end")
                self.ent_x._entradas[r][0].insert(0, str(v))
        except Exception:
            pass

        # Salida
        self.out = ttk.Frame(self); self.out.pack(fill="both", expand=True, padx=10, pady=10)
        self._registro: Registro = []

    def _actualizar(self):
        m, n = int(self.m.get()), int(self.n.get())
        self.ent_A.establecer_dimensiones(m, n)
        self.ent_x.establecer_dimensiones(n, 1)
        self.ent_y.establecer_dimensiones(n, 1)

    def _calcular(self):
        try:
            log: Registro = []

            A = self.ent_A.obtener_matriz()
            x = self.ent_x.obtener_matriz()
            y = self.ent_y.obtener_matriz()

            # (Ax)^T  y  x^T A^T
            Ax = multiplicar_matrices(A, x, registro=log)
            AxT = transpuesta(Ax, registro=log)
            xT  = transpuesta(x, registro=log)
            AT  = transpuesta(A, registro=log)
            xT_AT = multiplicar_matrices(xT, AT, registro=log)

            # xx^T  y  x^T x
            xxT = multiplicar_matrices(x, xT, registro=log)
            xTx = multiplicar_matrices(xT, x, registro=log)  # escalar 1x1

            # y^T x (solo si y no está vacía y coincide dimensión)
            yTx = None
            if y and len(y) == len(x):
                yT = transpuesta(y, registro=log)
                yTx = multiplicar_matrices(yT, x, registro=log)

            # ¿Está definida A^T x^T?
            ok, detalle = es_multiplicable(AT, xT)
            definida = "Sí" if ok else "No"
            razon = detalle + " (Para que A^T x^T exista, A debe tener exactamente 1 fila)."

            # Mostrar
            for w in self.out.winfo_children(): w.destroy()
            mostrar_matriz(self.out, Ax,   titulo="Ax (m x 1)")
            mostrar_matriz(self.out, AxT,  titulo="(Ax)^T (1 x m)")
            mostrar_matriz(self.out, xT_AT, titulo="x^T A^T (1 x m)")
            mostrar_matriz(self.out, xxT,  titulo="x x^T (n x n)")
            mostrar_matriz(self.out, xTx,  titulo="x^T x (1 x 1, escalar)")
            if yTx is not None:
                mostrar_matriz(self.out, yTx, titulo="y^T x (1 x 1, escalar)")

            # Conclusiones (dos formatos)
            m_, n_ = len(A), len(A[0]) if A and A[0] else 0
            texto_castellano = (
                "Conclusión (castellano):\n"
                f"¢ (Ax)^T y x^T A^T coinciden numéricamente (propiedad (AB)^T = B^T A^T).\n"
                f"¢ x x^T es matriz simétrica de rango 1 (si x ¢ 0). x^T x es el tamano al cuadrado de x (producto interno).\n"
                f"¢ A^T x^T está {definida.lower()}. {razon}"
            )
            texto_comb_lin = (
                "Conclusión (combinación lineal):\n"
                "¢ Ax es la combinación lineal de las columnas de A con coeficientes dados por las entradas de x; "
                "por tanto (Ax)^T = x^T A^T es el mismo resultado visto como combinación de las filas de A^T.\n"
                "¢ x x^T es el operador proyección/outer product generado por x (matriz de todas las combinaciones a_i a_j); "
                "x^T x es la suma de cuadrados de los coeficientes de esa combinación (norma al cuadrado)."
            )
            _, current_rows = self.out.grid_size()  # filas ocupadas hasta ahora
            lbl1 = ttk.Label(self.out, text=texto_castellano, justify="left")
            lbl1.grid(row=current_rows, column=0, sticky="w", pady=(10, 4), columnspan=10)

            lbl2 = ttk.Label(self.out, text=texto_comb_lin, justify="left")
            lbl2.grid(row=current_rows + 1, column=0, sticky="w", pady=(0, 6), columnspan=10)


            self._registro = log
            self.btn_log.config(state="normal")
        except Exception as e:
            messagebox.showerror("Error", str(e), parent=self)
            self.btn_log.config(state="disabled")

    def _ver_log(self):
        abrir_ventana_proceso(self, self._registro, titulo="Proceso: ejercicio de transpuestas")


class VentanaSuma(ttk.Frame):
    """Vista para sumar varias matrices del mismo tamano."""

    def __init__(self, maestro):
        super().__init__(maestro, padding=12)
        ttk.Label(self, text="Suma de matrices", font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 8))
        self.cantidad = tk.IntVar(value=2)

        controles = ttk.Frame(self)
        controles.pack(fill="x", pady=(0, 10))
        ttk.Label(controles, text="Numero de matrices:").pack(side="left")
        ttk.Spinbox(controles, from_=2, to=6, width=5, textvariable=self.cantidad).pack(side="left", padx=(4, 12))
        ttk.Button(controles, text="Aplicar", command=self._reconstruir_paneles).pack(side="left")
        ttk.Button(controles, text="Actualizar dimensiones", command=self._aplicar_dimensiones).pack(side="left", padx=(8, 0))
        ttk.Button(controles, text="Calcular suma", command=self._calcular).pack(side="left", padx=(20, 0))
        self.boton_registro_suma = ttk.Button(controles, text="Ver proceso", command=self._mostrar_proceso, state="disabled")
        self.boton_registro_suma.pack(side="left", padx=(8, 0))

        self.contenedor_matrices = ttk.Frame(self)
        self.contenedor_matrices.pack(fill="both", expand=False, pady=10)
        self.paneles: list[PanelMatriz] = []
        self._reconstruir_paneles()

        self.marco_resultado = ttk.Frame(self)
        self.marco_resultado.pack(fill="both", expand=True, pady=(0, 10))
        self._registro_final: Registro = []

    def _reconstruir_paneles(self) -> None:
        for widget in self.contenedor_matrices.winfo_children():
            widget.destroy()
        self.paneles = []
        for indice in range(int(self.cantidad.get())):
            panel = PanelMatriz(self.contenedor_matrices, indice + 1, 2, 2, prefijo_titulo="Matriz")
            panel.pack(fill="x", padx=6, pady=6)
            self.paneles.append(panel)

    def _aplicar_dimensiones(self) -> None:
        for panel in self.paneles:
            panel.aplicar_dimensiones()

    def _calcular(self) -> None:
        try:
            matrices = [panel.obtener_matriz() for panel in self.paneles]
            registro: Registro = []
            resultado = sumar_matrices_lista(matrices, registro=registro)
            for widget in self.marco_resultado.winfo_children():
                widget.destroy()
            mostrar_matriz(self.marco_resultado, resultado, titulo="Resultado suma")
            self._registro_final = registro
            self.boton_registro_suma.config(state="normal" if registro else "disabled")
        except Exception as error:
            messagebox.showerror("Error", str(error), parent=self)
            self.boton_registro_suma.config(state="disabled")

    def _mostrar_proceso(self) -> None:
        abrir_ventana_proceso(self, self._registro_final, titulo="Proceso: suma de matrices")


class VentanaResta(ttk.Frame):
    """Vista para restar matrices de forma secuencial."""

    def __init__(self, maestro):
        super().__init__(maestro, padding=12)
        ttk.Label(self, text="Resta de matrices", font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 8))
        self.cantidad = tk.IntVar(value=2)

        controles = ttk.Frame(self)
        controles.pack(fill="x", pady=(0, 10))
        ttk.Label(controles, text="Numero de matrices:").pack(side="left")
        ttk.Spinbox(controles, from_=2, to=6, width=5, textvariable=self.cantidad).pack(side="left", padx=(4, 12))
        ttk.Button(controles, text="Aplicar", command=self._reconstruir_paneles).pack(side="left")
        ttk.Button(controles, text="Actualizar dimensiones", command=self._aplicar_dimensiones).pack(side="left", padx=(8, 0))
        ttk.Button(controles, text="Calcular resta", command=self._calcular).pack(side="left", padx=(20, 0))
        self.boton_registro_resta = ttk.Button(controles, text="Ver proceso", command=self._mostrar_proceso, state="disabled")
        self.boton_registro_resta.pack(side="left", padx=(8, 0))

        self.contenedor_matrices = ttk.Frame(self)
        self.contenedor_matrices.pack(fill="both", expand=False, pady=10)
        self.paneles: list[PanelMatriz] = []
        self._reconstruir_paneles()

        self.marco_resultado = ttk.Frame(self)
        self.marco_resultado.pack(fill="both", expand=True, pady=(0, 10))
        self._registro_final: Registro = []

    def _reconstruir_paneles(self) -> None:
        for widget in self.contenedor_matrices.winfo_children():
            widget.destroy()
        self.paneles = []
        for indice in range(int(self.cantidad.get())):
            panel = PanelMatriz(self.contenedor_matrices, indice + 1, 2, 2, prefijo_titulo="Matriz")
            panel.pack(fill="x", padx=6, pady=6)
            self.paneles.append(panel)

    def _aplicar_dimensiones(self) -> None:
        for panel in self.paneles:
            panel.aplicar_dimensiones()

    def _calcular(self) -> None:
        try:
            matrices = [panel.obtener_matriz() for panel in self.paneles]
            registro: Registro = []
            resultado = restar_matrices_secuencial(matrices, registro=registro)
            for widget in self.marco_resultado.winfo_children():
                widget.destroy()
            mostrar_matriz(self.marco_resultado, resultado, titulo="Resultado resta")
            self._registro_final = registro
            self.boton_registro_resta.config(state="normal" if registro else "disabled")
        except Exception as error:
            messagebox.showerror("Error", str(error), parent=self)
            self.boton_registro_resta.config(state="disabled")

    def _mostrar_proceso(self) -> None:
        abrir_ventana_proceso(self, self._registro_final, titulo="Proceso: resta de matrices")


class VentanaMatrizInvertible(ttk.Frame):
    """Vista para comprobar invertibilidad y mostrar caracterizaciones."""

    def __init__(self, maestro):
        super().__init__(maestro, padding=12)
        ttk.Label(self, text="Matriz invertible (teorema)", font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 8))
        self.n = tk.IntVar(value=3)

        barra = ttk.Frame(self)
        barra.pack(fill="x", pady=(0, 10))
        ttk.Label(barra, text="Orden n:").pack(side="left")
        ttk.Spinbox(barra, from_=1, to=8, width=5, textvariable=self.n).pack(side="left", padx=(4, 12))
        ttk.Button(barra, text="Actualizar", command=self._actualizar).pack(side="left")
        ttk.Button(barra, text="Comprobar invertibilidad", command=self._calcular).pack(side="left", padx=(10, 0))
        self.btn_log = ttk.Button(barra, text="Ver proceso", command=self._ver_log, state="disabled")
        self.btn_log.pack(side="left", padx=(8, 0))

        panel = ttk.Frame(self)
        panel.pack(fill="x", pady=10)
        self.ent_A = EntradaMatriz(panel, 3, 3, titulo="Matriz A (n x n)")
        self.ent_A.pack(side="left", padx=10)

        self.out = ttk.Frame(self)
        self.out.pack(fill="both", expand=True)
        self._registro: Registro = []

    def _actualizar(self) -> None:
        n = int(self.n.get())
        self.ent_A.establecer_dimensiones(n, n)

    def _calcular(self) -> None:
        try:
            A = self.ent_A.obtener_matriz()
            log: Registro = []
            for widget in self.out.winfo_children():
                widget.destroy()

            mostrar_matriz(self.out, A, titulo="Matriz A")
            props = caracterizaciones_invertible(A, registro=log)
            invertible = props.get("a", False)

            invA = None
            try:
                invA = inversa_gauss_jordan(A, registro=log)
                mostrar_matriz(self.out, invA, titulo="A^{-1}")
                mostrar_matriz(self.out, multiplicar_matrices(A, invA), titulo="A * A^{-1}")
                mostrar_matriz(self.out, multiplicar_matrices(invA, A), titulo="A^{-1} * A")
            except Exception as err:
                ttk.Label(self.out, text=f"A no es invertible: {err}", foreground="red").pack(anchor="w", pady=(8, 0))

            resumen = ttk.Frame(self.out)
            resumen.pack(fill="x", pady=(10, 0))
            ttk.Label(resumen, text="Teorema de la matriz invertible:", font=("Segoe UI", 11, "bold")).pack(anchor="w")
            descripciones = {
                "a": "A es una matriz invertible",
                "b": "A es equivalente por filas a I_n",
                "c": "A tiene n posiciones pivote",
                "d": "El sistema Ax = 0 solo tiene la solucion trivial",
                "e": "Las columnas de A son linealmente independientes",
                "f": "La transformacion x -> Ax es inyectiva",
                "g": "Ax = b tiene solucion para todo b",
                "h": "Las columnas de A generan R^n",
                "i": "La transformacion x -> Ax es sobre R^n",
                "j": "Existe C con CA = I",
                "k": "Existe D con AD = I",
                "l": "A^T es invertible",
            }
            for clave in descripciones:
                if clave in props:
                    estado = "Si" if props[clave] else "No"
                    ttk.Label(resumen, text=f"({clave}) {descripciones[clave]}: {estado}").pack(anchor="w")

            texto_estado = "A es invertible." if invertible and invA is not None else "A no es invertible."
            ttk.Label(self.out, text=texto_estado, font=("Segoe UI", 11, "bold"), foreground="green" if invertible and invA is not None else "red").pack(anchor="w", pady=(8, 0))

            self._registro = log
            self.btn_log.config(state="normal" if log else "disabled")
        except Exception as error:
            messagebox.showerror("Error", str(error), parent=self)
            self.btn_log.config(state="disabled")

    def _ver_log(self) -> None:
        abrir_ventana_proceso(self, self._registro, titulo="Proceso: inversa y equivalencias")


class VentanaProducto(ttk.Frame):
    """Vista para multiplicar dos matrices."""

    def __init__(self, maestro):
        super().__init__(maestro, padding=12)
        ttk.Label(self, text="Multiplicacion de matrices (A * B)", font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 8))
        self.filas_a = tk.IntVar(value=2)
        self.columnas_a = tk.IntVar(value=2)
        self.columnas_b = tk.IntVar(value=2)

        controles = ttk.Frame(self)
        controles.pack(fill="x", pady=(0, 10))
        ttk.Label(controles, text="Filas A:").pack(side="left")
        ttk.Spinbox(controles, from_=1, to=10, width=5, textvariable=self.filas_a).pack(side="left", padx=(4, 12))
        ttk.Label(controles, text="Columnas A / Filas B:").pack(side="left")
        ttk.Spinbox(controles, from_=1, to=10, width=5, textvariable=self.columnas_a).pack(side="left", padx=(4, 12))
        ttk.Label(controles, text="Columnas B:").pack(side="left")
        ttk.Spinbox(controles, from_=1, to=10, width=5, textvariable=self.columnas_b).pack(side="left", padx=(4, 12))
        ttk.Button(controles, text="Actualizar", command=self._actualizar_dimensiones).pack(side="left", padx=(8, 0))

        marcos = ttk.Frame(self)
        marcos.pack(fill="x", pady=10)
        self.entrada_a = EntradaMatriz(marcos, 2, 2, titulo="Matriz A")
        self.entrada_a.pack(side="left", padx=10)
        self.entrada_b = EntradaMatriz(marcos, 2, 2, titulo="Matriz B")
        self.entrada_b.pack(side="left", padx=10)

        acciones = ttk.Frame(self)
        acciones.pack(fill="x", pady=(0, 10))
        ttk.Button(acciones, text="Calcular producto (A * B)", command=self._calcular).pack(side="left")
        self.boton_registro_producto = ttk.Button(acciones, text="Ver proceso", command=self._mostrar_proceso, state="disabled")
        self.boton_registro_producto.pack(side="left", padx=(8, 0))

        self.marco_resultado = ttk.Frame(self)
        self.marco_resultado.pack(fill="both", expand=True)
        self._registro_final: Registro = []

    def _actualizar_dimensiones(self) -> None:
        filas_a = int(self.filas_a.get())
        columnas_a = int(self.columnas_a.get())
        columnas_b = int(self.columnas_b.get())
        self.entrada_a.establecer_dimensiones(filas_a, columnas_a)
        self.entrada_b.establecer_dimensiones(columnas_a, columnas_b)

    def _calcular(self) -> None:
        try:
            A = self.entrada_a.obtener_matriz()
            B = self.entrada_b.obtener_matriz()
            registro: Registro = []
            resultado = multiplicar_matrices(A, B, registro=registro)
            for widget in self.marco_resultado.winfo_children():
                widget.destroy()
            mostrar_matriz(self.marco_resultado, resultado, titulo="Resultado (A * B)")
            self._registro_final = registro
            self.boton_registro_producto.config(state="normal" if registro else "disabled")
        except Exception as error:
            messagebox.showerror("Error", str(error), parent=self)
            self.boton_registro_producto.config(state="disabled")

    def _mostrar_proceso(self) -> None:
        abrir_ventana_proceso(self, self._registro_final, titulo="Proceso: producto de matrices")


class VentanaProductoCadena(ttk.Frame):
    """Vista para multiplicar una cadena de matrices."""

    def __init__(self, maestro):
        super().__init__(maestro, padding=12)
        ttk.Label(self, text="Producto encadenado de matrices", font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 8))
        self.cantidad = tk.IntVar(value=2)
        self.enlazar_dimensiones = tk.BooleanVar(value=True)

        controles = ttk.Frame(self)
        controles.pack(fill="x", pady=(0, 10))
        ttk.Label(controles, text="Numero de matrices:").pack(side="left")
        ttk.Spinbox(controles, from_=2, to=6, width=5, textvariable=self.cantidad).pack(side="left", padx=(4, 12))
        ttk.Checkbutton(controles, text="Autoajustar dimensiones", variable=self.enlazar_dimensiones).pack(side="left")
        ttk.Button(controles, text="Aplicar", command=self._reconstruir_paneles).pack(side="left", padx=(8, 0))
        ttk.Button(controles, text="Actualizar dimensiones", command=self._aplicar_dimensiones).pack(side="left", padx=(8, 0))
        ttk.Button(controles, text="Calcular producto", command=self._calcular).pack(side="left", padx=(20, 0))
        self.boton_registro_cadena = ttk.Button(controles, text="Ver proceso", command=self._mostrar_proceso, state="disabled")
        self.boton_registro_cadena.pack(side="left", padx=(8, 0))

        self.contenedor_matrices = ttk.Frame(self)
        self.contenedor_matrices.pack(fill="both", expand=False, pady=10)
        self.paneles: list[PanelMatriz] = []
        self._reconstruir_paneles()

        self.marco_resultado = ttk.Frame(self)
        self.marco_resultado.pack(fill="both", expand=True)
        self._registro_final: Registro = []

    def _reconstruir_paneles(self) -> None:
        for widget in self.contenedor_matrices.winfo_children():
            widget.destroy()
        self.paneles = []
        for indice in range(int(self.cantidad.get())):
            panel = PanelMatriz(self.contenedor_matrices, indice + 1, 2, 2, prefijo_titulo="Matriz")
            panel.pack(fill="x", padx=6, pady=6)
            self.paneles.append(panel)
        self._aplicar_enlace()

    def _aplicar_enlace(self) -> None:
        if not self.enlazar_dimensiones.get():
            return
        for indice in range(len(self.paneles) - 1):
            izq = self.paneles[indice]
            der = self.paneles[indice + 1]
            der.filas.set(int(izq.columnas.get()))

    def _aplicar_dimensiones(self) -> None:
        self._aplicar_enlace()
        for panel in self.paneles:
            panel.aplicar_dimensiones()

    def _calcular(self) -> None:
        try:
            matrices = [panel.obtener_matriz() for panel in self.paneles]
            registro: Registro = []
            resultado = multiplicar_cadena(matrices, registro=registro)
            for widget in self.marco_resultado.winfo_children():
                widget.destroy()
            mostrar_matriz(self.marco_resultado, resultado, titulo="Resultado producto")
            self._registro_final = registro
            self.boton_registro_cadena.config(state="normal" if registro else "disabled")
        except Exception as error:
            messagebox.showerror("Error", str(error), parent=self)
            self.boton_registro_cadena.config(state="disabled")

    def _mostrar_proceso(self) -> None:
        abrir_ventana_proceso(self, self._registro_final, titulo="Proceso: producto encadenado")


class VentanaDeterminante(ttk.Frame):
    """Vista para calcular el determinante de una matriz."""

    def __init__(self, maestro):
        super().__init__(maestro, padding=12)
        ttk.Label(self, text="Determinante por cofactores", font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 8))
        self.n = tk.IntVar(value=3)

        controles = ttk.Frame(self)
        controles.pack(fill="x", pady=(0, 10))
        ttk.Label(controles, text="Orden n:").pack(side="left")
        ttk.Spinbox(controles, from_=1, to=8, width=5, textvariable=self.n).pack(side="left", padx=(4, 12))
        ttk.Button(controles, text="Actualizar", command=self._actualizar).pack(side="left")
        ttk.Button(controles, text="Calcular det(A)", command=self._calcular).pack(side="left", padx=(10, 0))
        self.btn_log = ttk.Button(controles, text="Ver proceso", command=self._ver_log, state="disabled")
        self.btn_log.pack(side="left", padx=(8, 0))

        panel = ttk.Frame(self)
        panel.pack(fill="x", pady=10)
        self.ent_A = EntradaMatriz(panel, 3, 3, titulo="Matriz A (n x n)")
        self.ent_A.pack(side="left", padx=10)

        self.resultado_var = tk.StringVar(value="det(A) = 0")
        ttk.Label(self, textvariable=self.resultado_var, font=("Segoe UI", 12, "bold"), foreground="blue").pack(anchor="w")

        self._registro: Registro = []

    def _actualizar(self) -> None:
        n = int(self.n.get())
        self.ent_A.establecer_dimensiones(n, n)

    def _calcular(self) -> None:
        try:
            A = self.ent_A.obtener_matriz()
            registro: Registro = []
            det_valor = determinante(A, registro=registro)
            self.resultado_var.set(f"det(A) = {formatear_numero(det_valor)}")
            self._registro = registro
            self.btn_log.config(state="normal" if registro else "disabled")
        except Exception as error:
            messagebox.showerror("Error", str(error), parent=self)
            self.btn_log.config(state="disabled")

    def _ver_log(self) -> None:
        abrir_ventana_proceso(self, self._registro, titulo="Proceso: determinante (cofactores)")


class VentanaCramer(ttk.Frame):
    """Vista para resolver Ax = b con la Regla de Cramer."""

    def __init__(self, maestro):
        super().__init__(maestro, padding=12)
        ttk.Label(self, text="Regla de Cramer (Ax = b)", font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 8))
        self.n = tk.IntVar(value=3)

        controles = ttk.Frame(self)
        controles.pack(fill="x", pady=(0, 10))
        ttk.Label(controles, text="Orden n:").pack(side="left")
        ttk.Spinbox(controles, from_=1, to=8, width=5, textvariable=self.n).pack(side="left", padx=(4, 12))
        ttk.Button(controles, text="Actualizar", command=self._actualizar).pack(side="left")
        ttk.Button(controles, text="Resolver Ax = b", command=self._resolver).pack(side="left", padx=(10, 0))
        self.btn_log = ttk.Button(controles, text="Ver proceso", command=self._ver_log, state="disabled")
        self.btn_log.pack(side="left", padx=(8, 0))

        paneles = ttk.Frame(self)
        paneles.pack(fill="x", pady=10)
        self.ent_A = EntradaMatriz(paneles, 3, 3, titulo="Matriz A (n x n)")
        self.ent_A.pack(side="left", padx=10)
        self.ent_b = EntradaMatriz(paneles, 3, 1, titulo="Vector b (n x 1)")
        self.ent_b.pack(side="left", padx=10)

        self.marco_resultado = ttk.Frame(self)
        self.marco_resultado.pack(fill="both", expand=True)
        self._registro: Registro = []

    def _actualizar(self) -> None:
        n = int(self.n.get())
        self.ent_A.establecer_dimensiones(n, n)
        self.ent_b.establecer_dimensiones(n, 1)

    def _resolver(self) -> None:
        try:
            A = self.ent_A.obtener_matriz()
            b = self.ent_b.obtener_matriz()
            registro: Registro = []
            solucion = regla_cramer(A, b, registro=registro)
            for widget in self.marco_resultado.winfo_children():
                widget.destroy()
            mostrar_matriz(self.marco_resultado, solucion, titulo="Solucion x")
            self._registro = registro
            self.btn_log.config(state="normal" if registro else "disabled")
        except Exception as error:
            messagebox.showerror("Error", str(error), parent=self)
            self.btn_log.config(state="disabled")

    def _ver_log(self) -> None:
        abrir_ventana_proceso(self, self._registro, titulo="Proceso: Regla de Cramer")



class MenuLateral(ttk.Frame):
    """Menu lateral desplazable."""

    def __init__(
        self,
        maestro: tk.Misc,
        opciones: List[tuple[str, str]],
        on_select: Callable[[str], None],
        ancho: int = 240,
    ) -> None:
        super().__init__(maestro, width=ancho, padding=(12, 16))
        self.on_select = on_select
        self._botones: dict[str, ttk.Button] = {}
        self._crear_encabezado()
        self._crear_lista(opciones)

    def _crear_encabezado(self) -> None:
        cabecera = ttk.Frame(self)
        cabecera.pack(fill="x", pady=(0, 12))

        icono = tk.Canvas(cabecera, width=26, height=18, highlightthickness=0)
        icono.pack(side="left")
        for indice in range(3):
            icono.create_rectangle(2, 3 + indice * 6, 24, 6 + indice * 6, fill="#555555", outline="")

        ttk.Label(cabecera, text="Opciones", font=("Segoe UI", 12, "bold")).pack(side="left", padx=(8, 0))

    def _crear_lista(self, opciones: List[tuple[str, str]]) -> None:
        contenedor = ttk.Frame(self)
        contenedor.pack(fill="both", expand=True)

        canvas = tk.Canvas(contenedor, highlightthickness=0)
        scrollbar = ttk.Scrollbar(contenedor, orient="vertical", command=canvas.yview)
        interior = ttk.Frame(canvas)

        interior.bind("<Configure>", lambda evento: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=interior, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set, width=210)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for texto, clave in opciones:
            boton = ttk.Button(interior, text=texto, command=lambda k=clave: self._seleccionar(k))
            boton.pack(fill="x", pady=4)
            self._botones[clave] = boton

    def _seleccionar(self, clave: str) -> None:
        if self.on_select:
            self.on_select(clave)

    def marcar_seleccion(self, clave: str) -> None:
        for k, boton in self._botones.items():
            if k == clave:
                boton.state(["disabled"])
            else:
                boton.state(["!disabled"])
    
    def mostrar_newton(self):
        # Limpiar frame principal
        for widget in self.main_frame.winfo_children():
            widget.destroy()
            
        # Cargar la vista de Newton
        vista = VistaNewton(self.main_frame)
        # vista.pack() ya se llama dentro del __init__ de la clase
        
        self._ocultar_menu() # Cerrar menú lateral si aplica


class Aplicacion(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Calculadora de matrices")
        self.geometry("360x560")
        self.minsize(320, 480)

        # --- AGREGA ESTA LÍNEA AQUÍ AL PRINCIPIO ---
        self.vista_actual = None  # Inicializamos la variable para evitar el error
        self._menu_visible = False
        self.vistas_def = [
            ("calculadora", "Calculadora", SimpleCalculator),
            ("suma", "Sumar matrices", VentanaSuma),
            ("resta", "Restar matrices", VentanaResta),
            ("producto_cadena", "Multiplicar matrices", VentanaProductoCadena),
            #("producto", "Multiplicar (A * B)", VentanaProducto),
            ("independencia", "Independencia lineal", VentanaIndependencia),
            ("transpuesta", "Transpuesta A^T", VentanaTranspuesta),
            #("ejercicio_transpuestas", "Ejercicio transpuestas", VentanaEjercicioTranspuestas),
            ("invertible", "Matriz invertible", VentanaMatrizInvertible),
            ("determinante", "Determinante (cofactores)", VentanaDeterminante),
            ("cramer", "Regla de Cramer", VentanaCramer),
            #Metodos numericos
            ("biseccion", "Metodo de Biseccion", VentanaBiseccion),
            ("regla_falsa", "Metodo de Regla Falsa", VentanaReglaFalsa),
            ("newton", "Metodo de Newton", VistaNewton),
            ("secante", "Metodo de la Secante", VentanaSecante),
        ]
        self.vistas_info = {
            clave: {"nombre": nombre, "constructor": constructor}
            for clave, nombre, constructor in self.vistas_def
        }
        self.vistas_instanciadas: dict[str, ttk.Frame] = {}
        self.actual: Optional[ttk.Frame] = None
        self.dimensiones_vistas = {
            "calculadora": (360, 560, 320, 480),
            "suma": (900, 700, 900, 700),
            "resta": (900, 700, 900, 700),
            "producto_cadena": (1000, 750, 1000, 750),
            "producto": (900, 650, 900, 650),
            "independencia": (960, 700, 960, 700),
            "transpuesta": (820, 620, 820, 620),
            "ejercicio_transpuestas": (1080, 780, 1080, 780),
            "invertible": (1080, 800, 1080, 800),
            "determinante": (600, 450, 600, 450),
            "cramer": (800, 600, 800, 600),
        
            "biseccion": (700, 550, 600, 450),
            "regla_falsa": (700, 550, 600, 450),
        }

        self._construir_layout()
        self.bind("<Configure>", self._recolocar_menu)
        self._mostrar_vista("calculadora")

    def _construir_layout(self) -> None:
        self.contenedor = ttk.Frame(self)
        self.contenedor.pack(fill="both", expand=True)

        self.topbar = ttk.Frame(self.contenedor, padding=(12, 8))
        self.topbar.pack(fill="x")

        self._crear_boton_menu()

        self.titulo_var = tk.StringVar(value=self.vistas_info["calculadora"]["nombre"])
        ttk.Label(self.topbar, textvariable=self.titulo_var, font=("Segoe UI", 12, "bold")).pack(side="left", padx=(10, 0))

        self.vista_wrapper = tk.Frame(self.contenedor, bd=2, relief="ridge")
        self.vista_wrapper.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.contenedor_vistas = ttk.Frame(self.vista_wrapper, padding=12)
        self.contenedor_vistas.pack(fill="both", expand=True)

        opciones_menu = [(nombre, clave) for clave, nombre, _ in self.vistas_def]
        self.menu_lateral = MenuLateral(self, opciones_menu, on_select=self._seleccionar_desde_menu)

    def _crear_boton_menu(self) -> None:
        fondo = self.cget("bg") or "#f0f0f0"
        self.boton_menu = tk.Canvas(
            self.topbar,
            width=32,
            height=26,
            highlightthickness=0,
            highlightbackground=fondo,
            bd=0,
            background=fondo,
            cursor="hand2",
        )
        for indice in range(3):
            self.boton_menu.create_rectangle(6, 6 + indice * 6, 26, 8 + indice * 6, fill="#444444", outline="#444444")
        self.boton_menu.pack(side="left")
        self.boton_menu.bind("<Button-1>", self._toggle_menu)

    def _seleccionar_desde_menu(self, clave: str) -> None:
        self._mostrar_vista(clave)
        self._ocultar_menu()

    def _mostrar_vista(self, clave: str) -> None:
        info = self.vistas_info.get(clave)
        if info is None:
            return
        if self.vista_actual is not None:
            self.vista_actual.pack_forget()
        vista = self.vistas_instanciadas.get(clave)
        if vista is None:
            constructor = info["constructor"]
            vista = constructor(self.contenedor_vistas)
            try:
                vista.configure(padding=12)
            except tk.TclError:
                pass
            self.vistas_instanciadas[clave] = vista
        vista.pack(fill="both", expand=True)
        self.vista_actual = vista
        self.titulo_var.set(info["nombre"])
        self.menu_lateral.marcar_seleccion(clave)
        self._aplicar_dimensiones(clave)

    def _toggle_menu(self, _evento=None) -> None:
        if self._menu_visible:
            self._ocultar_menu()
        else:
            self._mostrar_menu()

    def _mostrar_menu(self) -> None:
        if self._menu_visible:
            self._recolocar_menu()
            return
        self._menu_visible = True
        self.menu_lateral.lift()
        self._recolocar_menu()

    def _ocultar_menu(self) -> None:
        if not self._menu_visible:
            return
        self._menu_visible = False
        self.menu_lateral.place_forget()

    def _recolocar_menu(self, _evento=None) -> None:
        if not self._menu_visible:
            return
        self.update_idletasks()
        y_topbar = self.topbar.winfo_rooty() - self.winfo_rooty()
        alto_topbar = self.topbar.winfo_height()
        y_inicio = y_topbar + alto_topbar
        altura = max(0, self.winfo_height() - y_inicio)
        self.menu_lateral.place(x=0, y=y_inicio, width=240, height=altura)
        self.menu_lateral.lift()

    def _aplicar_dimensiones(self, clave: str) -> None:
        ancho, alto, min_ancho, min_alto = self.dimensiones_vistas.get(clave, self.dimensiones_vistas["calculadora"])
        self.geometry(f"{ancho}x{alto}")
        self.minsize(min_ancho, min_alto)
        if self._menu_visible:
            self.after_idle(self._recolocar_menu)


if __name__ == "__main__":
    aplicacion = Aplicacion()
    aplicacion.mainloop()



