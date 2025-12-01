import math
import re
from typing import Tuple, List, Dict, Any

def _preprocesar_expresion(expr: str) -> str:
    if not expr: return ""
    # 1. Normalización básica
    expr = expr.strip().lower() # Todo a minúsculas para evitar errores X vs x
    expr = expr.replace(",", ".")
    expr = expr.replace("^", "**")
    expr = expr.replace("sen", "sin") # Español a Inglés
    
    # 2. Multiplicación implícita inteligente (2x -> 2*x)
    # Inserta * entre número y letra (2x)
    expr = re.sub(r'(?<=\d)\s*(?=[a-z\(])', '*', expr)
    # Inserta * entre paréntesis y número/letra: )x -> )*x, )2 -> )*2
    expr = re.sub(r'(?<=\))\s*(?=[\d a-z\(])', '*', expr)
    # Inserta * entre letra y paréntesis: x( -> x*( (excepto funciones como sin, cos)
    # Para evitar romper "sin(", usamos un lookbehind negativo o lista blanca simplificada
    # Simplemente asumimos que si hay una letra suelta antes de (, es una variable
    expr = re.sub(r'(?<=[x])\s*(?=\()', '*', expr) 
    
    return expr

def _crear_contexto_seguro(valor_x: float):
    """Crea el diccionario de variables y funciones matemáticas."""
    return {
        # Variables
        "x": valor_x,
        "e": math.e,
        "pi": math.pi,
        # Funciones
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "sqrt": math.sqrt,
        "exp": math.exp,
        "ln": math.log,
        "log": math.log10,
        "log10": math.log10,
        "abs": abs,
        "pow": pow
    }

def evaluar_funcion(func_str: str, val_x: float) -> float:
    func_py = _preprocesar_expresion(func_str)
    ctx = _crear_contexto_seguro(val_x)
    try:
        return float(eval(func_py, {"__builtins__": None}, ctx))
    except Exception as e:
        raise ValueError(f"Error evaluando '{func_str}' en x={val_x}: {e}")

def derivada_numerica(f_str: str, x: float, h=1e-5) -> float:
    f_x_h = evaluar_funcion(f_str, x + h)
    f_x_mh = evaluar_funcion(f_str, x - h)
    return (f_x_h - f_x_mh) / (2 * h)

# --- MÉTODOS ---

def newton_raphson(func_str: str, x0: float, tol=1e-7, max_iter=100):
    reg = []
    x = float(x0)
    for k in range(1, max_iter + 1):
        try:
            fx = evaluar_funcion(func_str, x)
            dfx = derivada_numerica(func_str, x)
        except ValueError:
            reg.append({'iter': k, 'xi': x, 'error': "Error Mat."})
            break

        if abs(dfx) < 1e-15:
            reg.append({'iter': k, 'xi': x, 'f(xi)': fx, 'error': "Derivada 0"})
            break
            
        x_new = x - fx / dfx
        error = abs(x_new - x)
        
        reg.append({'iter': k, 'xi': x, 'f(xi)': fx, 'error': error})
        
        if error < tol:
            return x_new, reg
        x = x_new
        
    return x, reg

def metodo_secante(func_str: str, x0: float, x1: float, tol=1e-7, max_iter=100):
    reg = []
    xa, xb = float(x0), float(x1)
    
    for k in range(1, max_iter + 1):
        try:
            fa = evaluar_funcion(func_str, xa)
            fb = evaluar_funcion(func_str, xb)
        except: break # Salir si eval falla
        
        if abs(fb - fa) < 1e-15:
            reg.append({'iter': k, 'xi': xb, 'error': "División por 0"})
            break
            
        xn = xb - fb * (xb - xa) / (fb - fa)
        error = abs(xn - xb)
        
        reg.append({'iter': k, 'xi': xb, 'xi+1': xn, 'error': error})
        
        if error < tol: return xn, reg
        xa, xb = xb, xn
        
    return xb, reg

def metodo_biseccion(func_str: str, a: float, b: float, tol=1e-7, max_iter=100):
    reg = []
    fa = evaluar_funcion(func_str, a)
    fb = evaluar_funcion(func_str, b)
    
    if fa * fb >= 0:
        raise ValueError("La función no cambia de signo en el intervalo [a, b].")
        
    c = a
    for k in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = evaluar_funcion(func_str, c)
        error = abs(b - a) / 2
        
        reg.append({'iter': k, 'a': a, 'b': b, 'c': c, 'error': error})
        
        if abs(fc) < 1e-15 or error < tol: return c, reg
        
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
            
    return c, reg

def metodo_regla_falsa(func_str: str, a: float, b: float, tol=1e-7, max_iter=100):
    reg = []
    fa = evaluar_funcion(func_str, a)
    fb = evaluar_funcion(func_str, b)
    
    if fa * fb >= 0: raise ValueError("Sin cambio de signo en [a, b].")
    
    c = a
    for k in range(1, max_iter + 1):
        if abs(fb - fa) < 1e-15: break
        
        c = (a*fb - b*fa) / (fb - fa)
        fc = evaluar_funcion(func_str, c)
        error = abs(c - a) # Estimación simple
        
        reg.append({'iter': k, 'a': a, 'b': b, 'c': c, 'error': error})
        
        if abs(fc) < 1e-15 or error < tol: return c, reg
        
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
            
    return c, reg