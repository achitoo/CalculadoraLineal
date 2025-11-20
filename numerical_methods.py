import math
import re
from fractions import Fraction
import numpy as np
from typing import Callable, List, Dict, Any, Optional, Tuple




RegistroBiseccion = List[Dict[str, Any]]

def _insertar_multiplicacion_implicita(expresion: str) -> str:
    """Inserta '*' donde el usuario omite la multiplicacion (p. ej. 3x -> 3*x)."""
    patrones = (
        r'(?<=[\d\.])\s*(?=[A-Za-z\(])',
        r'(?<=\))\s*(?=[\dA-Za-z\(])',
        r'(?<=x)\s*(?=\()',
    )
    for patron in patrones:
        expresion = re.sub(patron, '*', expresion)
    return expresion


def _preprocesar_expresion(expresion: str) -> str:
    """Normaliza la expresion del usuario antes de evaluarla."""
    expresion_py = expresion.replace('^', '**')
    return _insertar_multiplicacion_implicita(expresion_py)


def _crear_evaluador_frac(expresion: str) -> Callable[[Fraction], Fraction]:
    """
    Crea una función evaluable que opera con Fracciones a partir de un string.
    Maneja funciones de 'math' convirtiendo tipos F -> float -> F.
    """
    # Reemplaza el operador ^ y resuelve multiplicaciones implicitas
    expresion_py = _preprocesar_expresion(expresion)

    # Funciones que SÍ soportan Fraction directamente
    contexto_seguro = {
        'abs': abs,
    }
    
    # Funciones de 'math' que necesitan conversión float <-> Fraction
    # Creamos un "wrapper" para cada una
    for nombre_f in (
        'sqrt', 'sin', 'cos', 'tan', 'exp', 'log', 'log10', 
        'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh'
    ):
        if hasattr(math, nombre_f):
            # Esta lambda captura la función 'f_math' correcta
            contexto_seguro[nombre_f] = (
                lambda f_in, f_math=getattr(math, nombre_f): Fraction(str(f_math(float(f_in))))
            )
    
    # Constantes
    contexto_seguro['pi'] = Fraction(str(math.pi))
    contexto_seguro['e'] = Fraction(str(math.e))

    def evaluador(x_val: Fraction) -> Fraction:
        """
        La función interna que se llamará con cada valor de x.
        """
        try:
            # Preparamos el contexto local para esta evaluacion
            contexto_local = contexto_seguro.copy()
            contexto_local['x'] = x_val
            
            # Evaluar la expresión de forma segura
            # Usamos un diccionario de 'builtins' vacío para prevenir acceso a funciones peligrosas
            resultado = eval(expresion_py, {"__builtins__": {}}, contexto_local)
            
            # Asegurarse de que el resultado siempre sea Fraction
            if not isinstance(resultado, Fraction):
                return Fraction(str(resultado))
            return resultado
        
        except Exception as e:
            # Captura errores comunes (ej. "log(-1)") o errores de sintaxis
            raise ValueError(f"Error al evaluar f(x)='{expresion_py}' con x={formatear_numero_simple(x_val)}: {e}") from e

    return evaluador

def metodo_biseccion(
    expresion_f: str,
    a: Fraction,
    b: Fraction,
    tolerancia: Fraction,
    max_iter: int = 100
) -> Tuple[Fraction, RegistroBiseccion]:
    """
    Ejecuta el Método de Bisección usando Fracciones.
    
    Devuelve: (raiz_aproximada, registro_de_iteraciones)
    """
    
    registro: RegistroBiseccion = []
    
    try:
        # 1. Crear la función f(x) desde el string
        f = _crear_evaluador_frac(expresion_f)
    except Exception as e:
        raise ValueError(f"La expresión f(x) es invalida: {e}")

    # 2. Evaluar extremos del intervalo
    f_a = f(a)
    f_b = f(b)

    # 3. Validar condición de Bisección
    if f_a * f_b >= 0:
        # Comprobar si la raíz es exactamente un extremo
        if f_a == 0:
            registro.append({"iter": 0, "a": a, "f(a)": f_a, "b": b, "f(b)": f_b, "c": a, "f(c)": f_a, "error": 0})
            return a, registro
        if f_b == 0:
            registro.append({"iter": 0, "a": a, "f(a)": f_a, "b": b, "f(b)": f_b, "c": b, "f(c)": f_b, "error": 0})
            return b, registro
        
        # Si no, es un error
        raise ValueError(f"Error: f(a) y f(b) deben tener signos opuestos.\n\nf({formatear_numero_simple(a)}) = {formatear_numero_simple(f_a)}\nf({formatear_numero_simple(b)}) = {formatear_numero_simple(f_b)}")

    if a >= b:
        raise ValueError("El intervalo [a, b] es invalido (a debe ser menor que b).")

    c = a # Inicialización
    iteracion = 0
    
    # 4. Iniciar bucle de iteraciones
    while iteracion < max_iter:
        iteracion += 1
        
        c = (a + b) / 2      # Calcular punto medio (Bisección)
        f_c = f(c)           # Evaluar f(c)
        error_abs = abs(b - a) / 2 # Error actual
        
        # Guardar la fila para la tabla de proceso
        registro.append({
            "iter": iteracion, "a": a, "f(a)": f_a, "b": b, "f(b)": f_b,
            "c": c, "f(c)": f_c, "error": error_abs
        })

        # 5. Condición de parada
        if f_c == 0 or error_abs < tolerancia:
            return c, registro
        
        # 6. Redefinir el intervalo
        if f_a * f_c < 0:
            # La raíz está en [a, c]
            b = c
            f_b = f_c
        else:
            # La raíz está en [c, b]
            a = c
            f_a = f_c
    
    # 7. Si se alcanza max_iter, devolver la mejor aproximación
    return c, registro

def formatear_numero_simple(num: Fraction) -> str:
    """Formateador simple para mensajes de error, ya que no tenemos formatear_valor_ui aquí."""
    if num.denominator == 1:
        return str(num.numerator)
    return f"{num.numerator}/{num.denominator}"



def _crear_evaluador_numpy(expresion: str) -> Callable[[np.ndarray], np.ndarray]:
    """
    Crea una función evaluable que opera con arreglos de Numpy a partir de un string.
    """
    expresion_py = _preprocesar_expresion(expresion)

    contexto_seguro = {
        "np": np,
        "abs": np.abs,
        "sqrt": np.sqrt,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "log10": np.log10,
        "asin": np.arcsin,
        "acos": np.arccos,
        "atan": np.arctan,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
        "pi": np.pi,
        "e": np.e,
    }

    def evaluador(x_val: np.ndarray) -> np.ndarray:
        try:
            contexto_local = contexto_seguro.copy()
            contexto_local["x"] = x_val

            resultado = eval(expresion_py, {"__builtins__": {}}, contexto_local)

            if isinstance(resultado, (int, float, Fraction)):
                return np.full_like(x_val, float(resultado))

            return np.array(resultado, dtype=float)

        except Exception:
            if isinstance(x_val, np.ndarray):
                return np.full_like(x_val, np.nan)
            return np.nan

    return evaluador

def metodo_regla_falsa(
    expresion_f: str,
    a: Fraction,
    b: Fraction,
    tolerancia: Fraction,
    max_iter: int = 100
) -> Tuple[Fraction, RegistroBiseccion]:
    """
    Regla Falsa (Posición Falsa) con aritmética de Fraction y el mismo
    formato de registro que Bisección.
      - c = (a*f(b) - b*f(a)) / (f(b) - f(a))
      - Actualiza [a,b] según el signo de f(c)
      - Criterio de paro: |c - c_prev| < tolerancia  ó  f(c) == 0
    """
    registro: RegistroBiseccion = []

    # 1) Construir f(x)
    try:
        f = _crear_evaluador_frac(expresion_f)
    except Exception as e:
        raise ValueError(f"La expresión f(x) es inválida: {e}")

    if a >= b:
        raise ValueError("El intervalo [a, b] es inválido (a debe ser menor que b).")

    # 2) Evaluar extremos
    f_a = f(a)
    f_b = f(b)

    # Permite raíz exacta en extremos
    if f_a == 0:
        registro.append({"iter": 0, "a": a, "f(a)": f_a, "b": b, "f(b)": f_b, "c": a, "f(c)": f_a, "error": 0})
        return a, registro
    if f_b == 0:
        registro.append({"iter": 0, "a": a, "f(a)": f_a, "b": b, "f(b)": f_b, "c": b, "f(c)": f_b, "error": 0})
        return b, registro

    # Condición necesaria (como en Bisección)
    if f_a * f_b > 0:
        raise ValueError(
            "Error: f(a) y f(b) deben tener signos opuestos para la Regla Falsa.\n"
            f"f({a}) = {f_a}\n"
            f"f({b}) = {f_b}"
        )

    c_prev: Fraction | None = None
    c = a

    for k in range(1, max_iter + 1):
        denom = (f_b - f_a)
        if denom == 0:
            raise ValueError("f(a) y f(b) son iguales; no puede continuar la Regla Falsa (división entre cero).")

        # c = a - f(a)*(b-a)/(f(b)-f(a))   (forma equivalente exacta):
        c = (a * f_b - b * f_a) / denom
        f_c = f(c)

        error_abs: Fraction = abs(c - c_prev) if c_prev is not None else abs(b - a)

        registro.append({
            "iter": k, "a": a, "f(a)": f_a, "b": b, "f(b)": f_b,
            "c": c, "f(c)": f_c, "error": error_abs
        })

        if f_c == 0 or error_abs < tolerancia:
            return c, registro

        # Actualizar intervalo
        if f_a * f_c < 0:
            b, f_b = c, f_c
        else:
            a, f_a = c, f_c

        c_prev = c

    return c, registro

def derivada_numerica(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """
    Calcula la derivada aproximada de f en x usando diferencias centrales.
    f'(x) ≈ (f(x+h) - f(x-h)) / 2h
    """
    return (f(x + h) - f(x - h)) / (2 * h)

def newton_raphson(
    func_str: str,
    x0: float,
    tol: float = 1e-7,
    max_iter: int = 100
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Método de Newton-Raphson: x_{n+1} = x_n - f(x_n) / f'(x_n)
    
    Args:
        func_str: La función como string (ej: "x^2 - 4")
        x0: Valor inicial (semilla)
        tol: Tolerancia para el criterio de paro
        max_iter: Número máximo de iteraciones
        
    Returns:
        (Raíz aproximada, Lista de diccionarios con el historial)
    """
    # 1. Preparamos la función usando el preprocesador existente
    func_py = _preprocesar_expresion(func_str)
    
    # 2. Creamos un contexto matemático seguro y rápido para floats
    #    Usamos 'math' estándar en lugar de Fraction para velocidad en Newton
    contexto = vars(math).copy()
    contexto['x'] = 0.0
    contexto['e'] = math.e
    contexto['pi'] = math.pi
    # Aseguramos que si el usuario escribe "ln", funcione como "log"
    contexto['ln'] = math.log 

    def f(val_x: float) -> float:
        contexto['x'] = val_x
        try:
            # Evaluamos con float explícito
            return float(eval(func_py, {"__builtins__": None}, contexto))
        except Exception:
            return float('inf') # Retornar infinito si hay error matemático (ej. div por 0)

    registro = []
    x_actual = x0

    for k in range(1, max_iter + 1):
        # Calcular f(xi) y f'(xi)
        fx = f(x_actual)
        
        # Si encontramos la raíz exacta
        if fx == 0:
             registro.append({
                "iter": k, "xi": x_actual, "f(xi)": fx, 
                "f'(xi)": 0, "error": 0.0
            })
             return x_actual, registro

        dfx = derivada_numerica(f, x_actual)
        
        # Protección contra división por cero (derivada nula)
        if abs(dfx) < 1e-15:
            registro.append({
                "iter": k, "xi": x_actual, "f(xi)": fx, 
                "f'(xi)": dfx, "error": "Derivada ~ 0 (Punto estacionario)"
            })
            # No podemos continuar si la derivada es 0
            break
            
        # Fórmula de Newton: x_new = x_old - f(x)/f'(x)
        x_nuevo = x_actual - (fx / dfx)
        error = abs(x_nuevo - x_actual)
        
        registro.append({
            "iter": k,
            "xi": x_actual,
            "f(xi)": fx,
            "f'(xi)": dfx,
            "xi+1": x_nuevo,
            "error": error
        })

        # Criterio de parada
        if error < tol:
            return x_nuevo, registro
            
        x_actual = x_nuevo

    return x_actual, registro


