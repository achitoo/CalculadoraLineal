import re
from fractions import Fraction

def to_frac(x):
    try: return Fraction(str(x))
    except: return Fraction(0)

def parsear_matriz_texto(texto: str, max_filas=8, max_cols=8):
    """
    Convierte texto sucio (PDFs, LaTeX, Excel) a matriz numérica.
    """
    # 1. Normalización agresiva
    # Reemplazar barras invertidas y caracteres de lista por saltos
    texto = texto.replace("\\", "\n").replace(";", "\n").replace("(", " ").replace(")", " ")
    texto = texto.replace("[", " ").replace("]", " ").replace("{", " ").replace("}", " ")
    
    filas = []
    raw_rows = [r.strip() for r in texto.split('\n') if r.strip()]
    
    if len(raw_rows) > max_filas:
        # Si hay demasiadas filas, intentar ver si es un vector en una sola linea mal cortada
        pass 

    for r in raw_rows:
        # Extraer números (enteros, decimales, fracciones)
        # Soporta: -5, 3.4, 1/2
        numeros = re.findall(r'-?\d+(?:/\d+)?(?:\.\d+)?', r)
        if numeros:
            # Si hay demasiadas columnas, cortamos (o lanzamos error)
            if len(numeros) > max_cols:
                numeros = numeros[:max_cols]
            filas.append([to_frac(n) for n in numeros])
    
    # Validación final de dimensiones
    if not filas: return []
    return filas[:max_filas]

def parsear_sistema_ecuaciones(texto: str, max_vars=5):
    """
    Parsea ecuaciones lineales incluso con formato 'sucio'.
    Ej: "x+y=3 \ 2x-y=0"
    """
    # 1. Limpieza de separadores raros
    texto = texto.replace("\\", "\n").replace(";", "\n")
    
    lineas = []
    for l in texto.split('\n'):
        # Ignorar líneas que no parecen ecuaciones (ej: "Solución:")
        if not l.strip() or (('=' not in l) and not re.search(r'[a-zA-Z]', l)):
            continue
        lineas.append(l.strip())

    if not lineas: return [], [], []

    # 2. Identificar variables
    vars_found = set()
    patron_var = r'[a-zA-Z_]\w*' 
    
    for l in lineas:
        # Quitamos basura común de copiado
        clean_l = re.sub(r'^\d+[\.\)]', '', l) # Quita numeración "1." o "1)"
        lhs = clean_l.split('=')[0] if '=' in clean_l else clean_l
        vars_found.update(re.findall(patron_var, lhs))
    
    if not vars_found: return [], [], []

    # Orden inteligente: x,y,z o a,b,c o x1,x2...
    variables = sorted(list(vars_found))
    if {'x', 'y', 'z'}.issubset(set(variables)) and len(variables)<=3: variables = ['x','y','z'][:len(variables)]
    elif {'x', 'y'}.issubset(set(variables)): variables = ['x', 'y']
    
    A = []
    b = []

    # 3. Extraer coeficientes
    for l in lineas:
        l = re.sub(r'^\d+[\.\)]', '', l) # Limpiar numeración de ejercicio
        
        if '=' in l: lhs, rhs = l.split('=')
        else: lhs, rhs = l, "0"
            
        lhs = lhs.replace(" ", "")
        row_coeffs = {v: Fraction(0) for v in variables}
        
        # Regex: (+/-)(valor)(variable)
        terms = re.findall(r'([+\-]?)(\d+(?:/\d+)?(?:\.\d+)?)?([a-zA-Z_]\w*)', lhs)
        
        for signo, val, var in terms:
            if var not in variables: continue
            if val == "": num = Fraction(1)
            else: num = to_frac(val)
            if signo == "-": num *= -1
            row_coeffs[var] += num
            
        A.append([row_coeffs[v] for v in variables])
        
        # Lado derecho: limpiar basura no numérica
        try:
            rhs_clean = re.sub(r'[^0-9+\-*/\.\(\) ]', '', rhs)
            rhs_val = to_frac(eval(rhs_clean, {"__builtins__": None}))
        except:
            rhs_val = Fraction(0)
        b.append(rhs_val)

    return A, b, variables