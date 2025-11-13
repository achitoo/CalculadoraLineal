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
    Crea una funciÃ³n evaluable que opera con Fracciones a partir de un string.
    Maneja funciones de 'math' convirtiendo tipos F -> float -> F.
    """
    # Reemplaza el operador ^ y resuelve multiplicaciones implicitas
    expresion_py = _preprocesar_expresion(expresion)

    # Funciones que SÃ soportan Fraction directamente
    contexto_seguro = {
        'abs': abs,
    }
    
    # Funciones de 'math' que necesitan conversiÃ³n float <-> Fraction
    # Creamos un "wrapper" para cada una
    for nombre_f in (
        'sqrt', 'sin', 'cos', 'tan', 'exp', 'log', 'log10', 
        'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh'
    ):
        if hasattr(math, nombre_f):
            # Esta lambda captura la funciÃ³n 'f_math' correcta
            contexto_seguro[nombre_f] = (
                lambda f_in, f_math=getattr(math, nombre_f): Fraction(str(f_math(float(f_in))))
            )
    
    # Constantes
    contexto_seguro['pi'] = Fraction(str(math.pi))
    contexto_seguro['e'] = Fraction(str(math.e))

    def evaluador(x_val: Fraction) -> Fraction:
        """
        La funciÃ³n interna que se llamarÃ¡ con cada valor de x.
        """
        try:
            # Preparamos el contexto local para esta evaluaciÃ³n
            contexto_local = contexto_seguro.copy()
            contexto_local['x'] = x_val
            
            # Evaluar la expresiÃ³n de forma segura
            # Usamos un diccionario de 'builtins' vacÃ­o para prevenir acceso a funciones peligrosas
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
    Ejecuta el MÃ©todo de BisecciÃ³n usando Fracciones.
    
    Devuelve: (raiz_aproximada, registro_de_iteraciones)
    """
    
    registro: RegistroBiseccion = []
    
    try:
        # 1. Crear la funciÃ³n f(x) desde el string
        f = _crear_evaluador_frac(expresion_f)
    except Exception as e:
        raise ValueError(f"La expresiÃ³n f(x) es invÃ¡lida: {e}")

    # 2. Evaluar extremos del intervalo
    f_a = f(a)
    f_b = f(b)

    # 3. Validar condiciÃ³n de BisecciÃ³n
    if f_a * f_b >= 0:
        # Comprobar si la raÃ­z es exactamente un extremo
        if f_a == 0:
            registro.append({"iter": 0, "a": a, "f(a)": f_a, "b": b, "f(b)": f_b, "c": a, "f(c)": f_a, "error": 0})
            return a, registro
        if f_b == 0:
            registro.append({"iter": 0, "a": a, "f(a)": f_a, "b": b, "f(b)": f_b, "c": b, "f(c)": f_b, "error": 0})
            return b, registro
        
        # Si no, es un error
        raise ValueError(f"Error: f(a) y f(b) deben tener signos opuestos.\n\nf({formatear_numero_simple(a)}) = {formatear_numero_simple(f_a)}\nf({formatear_numero_simple(b)}) = {formatear_numero_simple(f_b)}")

    if a >= b:
        raise ValueError("El intervalo [a, b] es invÃ¡lido (a debe ser menor que b).")

    c = a # InicializaciÃ³n
    iteracion = 0
    
    # 4. Iniciar bucle de iteraciones
    while iteracion < max_iter:
        iteracion += 1
        
        c = (a + b) / 2      # Calcular punto medio (BisecciÃ³n)
        f_c = f(c)           # Evaluar f(c)
        error_abs = abs(b - a) / 2 # Error actual
        
        # Guardar la fila para la tabla de proceso
        registro.append({
            "iter": iteracion, "a": a, "f(a)": f_a, "b": b, "f(b)": f_b,
            "c": c, "f(c)": f_c, "error": error_abs
        })

        # 5. CondiciÃ³n de parada
        if f_c == 0 or error_abs < tolerancia:
            return c, registro
        
        # 6. Redefinir el intervalo
        if f_a * f_c < 0:
            # La raÃ­z estÃ¡ en [a, c]
            b = c
            f_b = f_c
        else:
            # La raÃ­z estÃ¡ en [c, b]
            a = c
            f_a = f_c
    
    # 7. Si se alcanza max_iter, devolver la mejor aproximaciÃ³n
    return c, registro

def formatear_numero_simple(num: Fraction) -> str:
    """Formateador simple para mensajes de error, ya que no tenemos formatear_valor_ui aquÃ­."""
    if num.denominator == 1:
        return str(num.numerator)
    return f"{num.numerator}/{num.denominator}"



def _crear_evaluador_numpy(expresion: str) -> Callable[[np.ndarray], np.ndarray]:
    """
    Crea una funciÃ³n evaluable que opera con arreglos de Numpy a partir de un string.
    """
    expresion_py = _preprocesar_expresion(expresion)

    # Contexto seguro con funciones de NUMPY
    contexto_seguro = {
        'np': np,
        'abs': np.abs,
        'sqrt': np.sqrt,
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'exp': np.exp,
        'log': np.log,
        'log10': np.log10,
        'asin': np.arcsin,
        'acos': np.arccos,
        'atan': np.arctan,
        'sinh': np.sinh,
        'cosh': np.cosh,
        'tanh': np.tanh,
        'pi': np.pi,
        'e': np.e,
    }

    def evaluador(x_val: np.ndarray) -> np.ndarray:
        """
        La funciÃ³n interna que se llamarÃ¡ con un arreglo de x.
        """
        try:
            contexto_local = contexto_seguro.copy()
            contexto_local['x'] = x_val

            resultado = eval(expresion_py, {"__builtins__": {}}, contexto_local)

            # Asegurar que la funciÃ³n devuelva un arreglo si el usuario solo puso un nÃºmero (ej. f(x) = 5)
            if isinstance(resultado, (int, float, Fraction)):
                return np.full_like(x_val, float(resultado))

            return np.array(resultado, dtype=float)

        except Exception as e:
            # Captura errores (ej. "log(-1)") y los silencia para el grÃ¡fico
            # Devuelve 'nan' (Not a Number) para que matplotlib no lo dibuje
            if isinstance(x_val, np.ndarray):
                return np.full_like(x_val, np.nan)
            return np.nan

    return evaluador

