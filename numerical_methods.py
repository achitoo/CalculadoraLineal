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

    registro: RegistroBiseccion = []

    # =====================================================
    # 1) Construir f(x)
    # =====================================================
    try:
        f = _crear_evaluador_frac(expresion_f)
    except Exception as e:
        raise ValueError(f"La expresión f(x) es inválida: {e}")

    # =====================================================
    # 2) Verificar intervalo
    # =====================================================
    if a >= b:
        raise ValueError("Intervalo inválido: se requiere que a < b.")

    # =====================================================
    # 3) Probar si la función realmente depende de x
    # =====================================================
    try:
        f_0 = f(Fraction(0))
        f_1 = f(Fraction(1))
        if f_0 == f_1:
            # Podría ser constante → revisar otros puntos
            f_2 = f(Fraction(2))
            if f_0 == f_2:
                raise ValueError(
                    "La función parece ser constante. "
                    "La Regla Falsa no puede aplicarse porque no existe un cambio de signo."
                )
    except:
        pass  # si no puede evaluar, no importa aquí

    # =====================================================
    # 4) Evaluar extremos
    # =====================================================
    try:
        f_a = f(a)
        f_b = f(b)
    except Exception as e:
        raise ValueError(f"No es posible evaluar f(x) en los extremos del intervalo: {e}")

    # Raíz exacta en extremos
    if f_a == 0:
        registro.append({"iter": 0, "a": a, "f(a)": f_a, "b": b, "f(b)": f_b, "c": a, "f(c)": f_a, "error": 0})
        return a, registro
    if f_b == 0:
        registro.append({"iter": 0, "a": a, "f(a)": f_a, "b": b, "f(b)": f_b, "c": b, "f(c)": f_b, "error": 0})
        return b, registro

    # =====================================================
    # 5) Verificar signo
    # =====================================================
    if f_a * f_b > 0:
        raise ValueError(
            "No se puede aplicar Regla Falsa: f(a) y f(b) tienen el MISMO signo.\n"
            f"f({a}) = {f_a}\n"
            f"f({b}) = {f_b}\n"
            "→ No existe garantía de raíz dentro del intervalo."
        )

    # =====================================================
    # 6) Detectar discontinuidades dentro del intervalo
    # =====================================================
    puntos_prueba = 10
    x_prev = a
    f_prev = f_a

    for i in range(1, puntos_prueba + 1):
        x_i = a + (b - a) * Fraction(i, puntos_prueba)
        try:
            f_i = f(x_i)
        except:
            raise ValueError(
                f"La función no es continua dentro del intervalo.\n"
                f"Error al evaluar f({x_i})."
            )

        # Discontinuidad evidente: salto enorme
        if abs(f_i - f_prev) > 1e6:
            raise ValueError(
                "La función presenta una discontinuidad dentro del intervalo.\n"
                "La Regla Falsa solo funciona con funciones continuas."
            )

        x_prev, f_prev = x_i, f_i

    # =====================================================
    # 7) Iteraciones normales (igual que tu versión)
    # =====================================================
    c_prev: Fraction | None = None
    c = a

    for k in range(1, max_iter + 1):
        denom = (f_b - f_a)
        if denom == 0:
            raise ValueError("Error: f(a) = f(b). No se puede continuar (división entre cero).")

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

    func_py = _preprocesar_expresion(func_str)

    contexto = vars(math).copy()
    contexto['x'] = 0.0
    contexto['e'] = math.e
    contexto['pi'] = math.pi
    contexto['ln'] = math.log 

    def f(val_x: float) -> float:
        contexto['x'] = val_x
        try:
            val = eval(func_py, {"__builtins__": None}, contexto)
            return float(val)
        except Exception:
            return float('inf')  # Discontinuidad o error interno

    registro = []
    x_actual = x0

    # Para detección de ciclos u oscilación
    ultimos_x = []

    for k in range(1, max_iter + 1):

        fx = f(x_actual)

        # ----------- (1) Discontinuidad cercana -----------
        if math.isinf(fx) or abs(fx) > 1e12:
            registro.append({
                "iter": k,
                "xi": x_actual,
                "f(xi)": fx,
                "f'(xi)": None,
                "error": "Discontinuidad detectada (|f(x)| → ∞)"
            })
            break

        # Si f(x) = 0, raíz exacta
        if fx == 0:
            registro.append({
                "iter": k, "xi": x_actual, "f(xi)": fx, 
                "f'(xi)": 0, "error": 0.0
            })
            return x_actual, registro

        # Derivada numérica
        dfx = derivada_numerica(f, x_actual)

        # ----------- (2) Función no derivable ------------
        if math.isinf(dfx) or math.isnan(dfx):
            registro.append({
                "iter": k,
                "xi": x_actual,
                "f(xi)": fx,
                "f'(xi)": dfx,
                "error": "Derivada indefinida o infinita"
            })
            break

        # ----------- (3) Derivada cero -------------------
        if abs(dfx) < 1e-15:
            registro.append({
                "iter": k,
                "xi": x_actual,
                "f(xi)": fx,
                "f'(xi)": dfx,
                "error": "Derivada casi 0 (punto estacionario)"
            })
            break

        # Fórmula de Newton
        x_nuevo = x_actual - (fx / dfx)
        error = abs(x_nuevo - x_actual)

        # Registrar
        registro.append({
            "iter": k,
            "xi": x_actual,
            "f(xi)": fx,
            "f'(xi)": dfx,
            "xi+1": x_nuevo,
            "error": error
        })

        # ----------- (4) Divergencia ----------------------
        if abs(x_nuevo) > 1e10:  # Número de magnitud absurda
            registro.append({
                "iter": k,
                "xi": x_actual,
                "f(xi)": fx,
                "f'(xi)": dfx,
                "error": "Divergencia: x_n crece demasiado"
            })
            break

        # ----------- (5) Ciclo / Oscilación ---------------
        ultimos_x.append(x_nuevo)
        if len(ultimos_x) > 6:  # conservar últimos 6
            ultimos_x.pop(0)
        
        # detectar repetición exacta o doble ciclo a↔b
        if len(ultimos_x) >= 3:
            # ciclo tipo a-b-a-b
            if len(set([round(val, 12) for val in ultimos_x])) <= 2:
                registro.append({
                    "iter": k,
                    "xi": x_actual,
                    "f(xi)": fx,
                    "f'(xi)": dfx,
                    "error": "Oscilación / Ciclo de Newton detectado"
                })
                break
        
        # ----------- (6) Stagnation -----------------------
        if error < tol:
            return x_nuevo, registro

        if error == 0:  # No está progresando
            registro.append({
                "iter": k,
                "xi": x_actual,
                "f(xi)": fx,
                "f'(xi)": dfx,
                "error": "Stagnation: no hay avance"
            })
            break

        x_actual = x_nuevo

    return x_actual, registro

def secante_falsaPosicion(
    func_str: str,
    x0: float,
    x1: float,
    tol: float = 1e-7,
    max_iter: int = 100
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Método de la Secante (Falsa Posición tipo secante) usando:
        x_{i+1} = x_i - f(x_i)*(x_{i-1} - x_i) / (f(x_{i-1}) - f(x_i))

    Args:
        func_str: f(x) como string
        x0: primer valor inicial  (x_{i-1})
        x1: segundo valor inicial (x_i)
        tol: tolerancia
        max_iter: número máximo de iteraciones

    Returns:
        (raíz aproximada, historial de iteraciones)
    """

    # Preparar función evaluable
    func_py = _preprocesar_expresion(func_str)
    contexto = vars(math).copy()
    contexto['e'] = math.e
    contexto['pi'] = math.pi
    contexto['ln'] = math.log

    def f(val_x: float) -> float:
        contexto['x'] = val_x
        try:
            return float(eval(func_py, {"__builtins__": None}, contexto))
        except Exception:
            return float('inf')  # discontinuidad

    registro = []
    xi_1 = x0
    xi = x1

    for k in range(1, max_iter + 1):

        f_xi_1 = f(xi_1)
        f_xi = f(xi)

        # ----------- Discontinuidades -----------
        if math.isinf(f_xi) or math.isinf(f_xi_1):
            registro.append({
                "iter": k,
                "xi-1": xi_1,
                "xi": xi,
                "f(xi-1)": f_xi_1,
                "f(xi)": f_xi,
                "error": "Discontinuidad detectada"
            })
            break

        # Raíz exacta
        if f_xi == 0:
            registro.append({
                "iter": k,
                "xi-1": xi_1,
                "xi": xi,
                "f(xi-1)": f_xi_1,
                "f(xi)": f_xi,
                "error": 0
            })
            return xi, registro

        denom = (f_xi_1 - f_xi)

        # ----------- División por cero -----------
        if abs(denom) < 1e-15:
            registro.append({
                "iter": k,
                "xi-1": xi_1,
                "xi": xi,
                "f(xi-1)": f_xi_1,
                "f(xi)": f_xi,
                "error": "Denominador casi 0 (puntos con misma f(x))"
            })
            break

        # Fórmula de la secante EXACTA de tu imagen
        x_nuevo = xi - f_xi * (xi_1 - xi) / denom
        error = abs(x_nuevo - xi)

        registro.append({
            "iter": k,
            "xi-1": xi_1,
            "xi": xi,
            "f(xi-1)": f_xi_1,
            "f(xi)": f_xi,
            "xi+1": x_nuevo,
            "error": error
        })

        # Criterio de parada
        if error < tol:
            return x_nuevo, registro

        # ----------- Detección de divergencia -----------
        if abs(x_nuevo) > 1e10:
            registro.append({
                "iter": k,
                "xi": xi,
                "xi+1": x_nuevo,
                "error": "Divergencia detectada"
            })
            break

        # ----------- Detección de oscilación -----------
        if abs(x_nuevo - xi_1) < tol and abs(xi - xi_1) < tol:
            registro.append({
                "iter": k,
                "xi": xi,
                "xi+1": x_nuevo,
                "error": "Oscilación detectada"
            })
            break

        # Mover valores para la siguiente iteración
        xi_1, xi = xi, x_nuevo

    return xi, registro
