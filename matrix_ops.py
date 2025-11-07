from typing import List, Tuple, Optional, Dict, Union, Sequence
from fractions import Fraction


Numero = Fraction
NumeroLike = Union[int, float, Fraction]
Matriz = List[List[Numero]]
RegistroEntrada = Union[str, Tuple[str, str, Matriz]]
Registro = List[RegistroEntrada]


def _a_fraccion(valor: NumeroLike) -> Fraction:
    """Intenta convertir el valor recibido a Fraction."""
    return valor if isinstance(valor, Fraction) else Fraction(valor)

def copiar_matriz(matriz: Sequence[Sequence[NumeroLike]]) -> Matriz:
    """Devuelve una copia independiente normalizando a Fraction cada entrada."""
    return [[_a_fraccion(valor) for valor in fila] for fila in matriz]


def registrar_matriz(
    registro: Optional[Registro],
    titulo: str,
    matriz: Sequence[Sequence[NumeroLike]],
) -> None:
    """Agrega al registro una entrada estructurada de tipo matriz."""
    if registro is not None:
        registro.append(("matrix", titulo, copiar_matriz(matriz)))

def rref(matriz: Matriz, registro: Optional[Registro] = None, tol: float = 1e-10) -> Tuple[Matriz, List[int]]:
    """Devuelve (R , Pivotes) Donde R es la forma reducida por filas y pivotes las columnas pivote (0-inde)"""
    _registrar(registro, "Cálculo de la forma reducida por filas (RREF)")
    if not matriz:
        return [] , []
    
    A= copiar_matriz(matriz)
    filas, cols = dimensiones_matriz(A)
    fila_piv =0
    pivotes : List[int] = []

    for col in range(cols):
        # buscamos pivote por valor absoluto maximo desde fila_piv
        piv = max(range(fila_piv, filas) , key = lambda f: abs(A[f][col]))
        if abs(A[piv][col]) <= tol:
            _registrar(registro, f"Columna {col+1}: sin pivote (columna dependiente / libre)")
            continue # columna sin pivote

        #intercambiamos fila_piv con piv si es necesario
        if piv != fila_piv:
            A[fila_piv] , A[piv] = A[piv] , A[fila_piv]
            _registrar(registro, f"Intercambio filas F{fila_piv+1} <--> F{piv+1}")
            registrar_matriz(registro, "A", A)

        # normalizamos fila_piv para que el pivote sea 1
        pivote = A[fila_piv][col]
        if abs(pivote - 1) > tol:
            for c in range(cols):
                A[fila_piv][c] /= pivote
            _registrar(registro, f"Normalizamos fila F{fila_piv+1} / {formatear_numero(pivote)}")
            registrar_matriz(registro, "A", A)

        #Hacer ceros en el resto de las filas
        for f in range (filas):
            if f == fila_piv:
                continue
            factor = A[f][col]
            if abs(factor) <= tol:
                continue # ya es cero
            for c in range(cols):
                A[f][c] -= factor * A[fila_piv][c]
            _registrar(registro, f"F{f+1} := F{f+1} - {formatear_numero(factor)} * F{fila_piv+1}")   #anular encima y debajo del pivote
            registrar_matriz(registro, "A", A)
        pivotes.append(col)
        fila_piv +=1
        if fila_piv == filas:
            break # ya no hay mas filas

    # Limpieza de -0.0 y valores muy pequeños
    for r in range(filas):
        for c in range(cols):
            if abs(A[r][c]) <= tol:
                A[r][c] = Fraction(0)

    registrar_matriz(registro, "RREF", A)
    return A, pivotes
    

def rango_matriz(matriz: Matriz, registro: Optional[Registro] = None, tol: float = 1e-10) -> Tuple[int, Matriz, List[int]]:
    """Devuelve (rango, RREF, pivotes)."""
    R, pivotes = rref(matriz, registro=registro, tol=tol)
    rango = len(pivotes)
    _registrar(registro, f"Rango = {rango}. Columnas pivote (1-index): " + ", ".join(str(p+1) for p in pivotes))
    return rango, R, pivotes

def formatear_numero(valor: Numero, decimales_maximos: int = 6) -> str:
    """Convierte un numero a cadena sin ceros sobrantes."""
    if abs(valor - round(valor)) < 10 ** (-(decimales_maximos + 1)):
        return str(int(round(valor)))
    cadena = f"{valor:.{decimales_maximos}f}"
    if "." in cadena:
        cadena = cadena.rstrip("0").rstrip(".")
    return cadena


def _registrar(registro: Optional[Registro], mensaje: str) -> None:
    if registro is not None:
        registro.append(mensaje)


def _formatear_matriz(matriz: Matriz, decimales_maximos: int = 6) -> str:
    if not matriz:
        return "[ ]"
    filas_texto = []
    for fila in matriz:
        texto_fila = ", ".join(formatear_numero(valor, decimales_maximos) for valor in fila)
        filas_texto.append(f"[ {texto_fila} ]")
    return "\n".join(filas_texto)


def dimensiones_matriz(matriz: Matriz) -> Tuple[int, int]:
    """Devuelve las dimensiones de la matriz."""
    if not matriz:
        return 0, 0
    return len(matriz), len(matriz[0]) if matriz[0] else 0


def _validar_rectangular(matriz: Matriz) -> None:
    """Verifica que todas las filas tengan la misma cantidad de columnas."""
    if not matriz:
        return
    columnas = len(matriz[0])
    for fila in matriz:
        if len(fila) != columnas:
            raise ValueError("La matriz no es rectangular")
#Funciones para resolver una determinante

def _get_submatrix(matriz: Matriz, row_skip: int, col_skip: int) -> Matriz:
    """Helper: Devuelve la submatriz eliminando una fila y una columna."""
    return [
        [matriz[i][j] for j in range(len(matriz[0])) if j != col_skip]
        for i in range(len(matriz)) if i != row_skip
    ]

def _determinante_recursivo(matriz: Matriz, registro: Optional[Registro] = None, nivel: int = 0) -> Numero:
    """Función recursiva interna para el determinante."""
    n = len(matriz)
    
    # Caso base 1x1
    if n == 1:
        return matriz[0][0]

    # Caso base 2x2
    if n == 2:
        det_2x2 = matriz[0][0] * matriz[1][1] - matriz[0][1] * matriz[1][0]
        return det_2x2

    # Expansión por cofactores (fila 0)
    det = Fraction(0)
    indent = "  " * nivel
    
    for j in range(n):
        elemento = matriz[0][j]
        # Optimización: si el elemento es 0, el cofactor no suma
        if abs(elemento) < 1e-10: 
            continue 
        
        signo = (-1) ** j
        sub_matriz = _get_submatrix(matriz, 0, j)

        if registro is not None:
            _registrar(registro, f"{indent}Columna {j+1}: a_1,{j+1} = {formatear_numero(elemento)}")

        det_sub = _determinante_recursivo(sub_matriz, registro, nivel + 1)  # Registrar sub-pasos
        cofactor = signo * det_sub
        termino = elemento * cofactor

        if registro is not None:
            if n > 2:
                _registrar(registro, f"{indent}  -> det(M_1,{j+1}) = {formatear_numero(det_sub)}")
            _registrar(registro, f"{indent}C_1,{j+1} = {formatear_numero(signo)} * det(M_1,{j+1}) = {formatear_numero(cofactor)}")
            _registrar(
                registro,
                f"{indent}Termino columna {j+1}: a_1,{j+1} * C_1,{j+1} = {formatear_numero(elemento)} * {formatear_numero(cofactor)} = {formatear_numero(termino)}",
            )

        det += termino

    if registro is not None and n > 2:
         _registrar(registro, f"{indent}Suma (Fila 1) = {formatear_numero(det)}")
    
    return det

def determinante(matriz: Matriz, registro: Optional[Registro] = None, tol: float = 1e-10) -> Numero:
    """
    Calcula el determinante de una matriz cuadrada por expansión de cofactores.
    """
    _registrar(registro, "Cálculo del determinante por cofactores")
    _validar_rectangular(matriz)
    n, m = dimensiones_matriz(matriz)
    if n != m:
        raise ValueError(f"La matriz debe ser cuadrada (es {n}x{m})")
    if n == 0:
        return Fraction(1)  # Determinante de matriz vacía (convención)
    
    registrar_matriz(registro, "A", matriz)

    # Usar la versión interna recursiva
    return _determinante_recursivo(matriz, registro=registro, nivel=0)
# Resolver una determinante mediante regla de cramer

def regla_cramer(A: Matriz, b_vec: Matriz, registro: Optional[Registro] = None, tol: float = 1e-10) -> Matriz:
    """
    Resuelve Ax = b usando la Regla de Cramer.
    A debe ser (n x n) y b_vec (n x 1).
    Devuelve x como un vector columna (n x 1).
    """
    _registrar(registro, "Resolución por Regla de Cramer: Ax = b")
    _validar_rectangular(A)
    _validar_rectangular(b_vec)
    
    n, m = dimensiones_matriz(A)
    nb, mb = dimensiones_matriz(b_vec)

    if n != m:
        raise ValueError(f"La matriz A debe ser cuadrada (es {n}x{m})")
    if n == 0:
        return []
    if nb != n or mb != 1:
        raise ValueError(f"El vector b debe ser un vector columna {n}x1 (es {nb}x{mb})")

    registrar_matriz(registro, "A", A)
    registrar_matriz(registro, "b", b_vec)

    # Extraer b como lista simple
    b_lista = [b_vec[i][0] for i in range(n)]

    # 1. Determinante de A
    _registrar(registro, "\nPASO 1: Calcular det(A)")
    det_A = determinante(A, registro=registro)
    _registrar(registro, f"-> det(A) = {formatear_numero(det_A)}")

    if abs(det_A) < tol:
        raise ValueError("El sistema no tiene solución única (det(A) = 0)")

    soluciones = []
    
    # 2. Calcular det(Ai) para cada columna i
    for i in range(n):
        variable_nombre = f"x_{i+1}"
        _registrar(registro, f"\nPASO {i+2}: Calcular det(A_{i+1}) para {variable_nombre}")
        
        # Crear Ai (copiando A)
        Ai = copiar_matriz(A)
        # Reemplazar columna i
        for fila_idx in range(n):
            Ai[fila_idx][i] = b_lista[fila_idx]
        
        registrar_matriz(registro, f"A_{i+1} (A con columna {i+1} reemplazada por b)", Ai)
        
        det_Ai = determinante(Ai, registro=registro)
        _registrar(registro, f"-> det(A_{i+1}) = {formatear_numero(det_Ai)}")

        sol = det_Ai / det_A
        soluciones.append(sol)
        _registrar(registro, f"SOLUCIÓN {variable_nombre} = det(A_{i+1}) / det(A) = {formatear_numero(det_Ai)} / {formatear_numero(det_A)} = {formatear_numero(sol)}")

    # Devolver como vector columna
    resultado = vector_columna(soluciones)
    registrar_matriz(registro, "Resultado", resultado)
    return resultado


def sumar_matrices_dos(matriz_a: Matriz, matriz_b: Matriz, registro: Optional[Registro] = None) -> Matriz:
    """Suma elemento a elemento dos matrices del mismo tamano."""
    _registrar(registro, "Suma de matrices: C = A + B")
    _validar_rectangular(matriz_a)
    _validar_rectangular(matriz_b)
    filas_a, columnas_a = dimensiones_matriz(matriz_a)
    filas_b, columnas_b = dimensiones_matriz(matriz_b)
    if filas_a != filas_b or columnas_a != columnas_b:
        raise ValueError("Para sumar se necesitan matrices con identicas dimensiones")
    _registrar(registro, f"Dimensiones: A = {filas_a}x{columnas_a}, B = {filas_b}x{columnas_b}")
    registrar_matriz(registro, "A", matriz_a)
    registrar_matriz(registro, "B", matriz_b)
    resultado = [
        [matriz_a[fila][columna] + matriz_b[fila][columna] for columna in range(columnas_a)]
        for fila in range(filas_a)
    ]
    registrar_matriz(registro, "Resultado", resultado)
    return resultado


def restar_matrices_dos(matriz_a: Matriz, matriz_b: Matriz, registro: Optional[Registro] = None) -> Matriz:
    """Resta elemento a elemento dos matrices del mismo tamano."""
    _registrar(registro, "Resta de matrices: C = A - B")
    _validar_rectangular(matriz_a)
    _validar_rectangular(matriz_b)
    filas_a, columnas_a = dimensiones_matriz(matriz_a)
    filas_b, columnas_b = dimensiones_matriz(matriz_b)
    if filas_a != filas_b or columnas_a != columnas_b:
        raise ValueError("Para restar se necesitan matrices con identicas dimensiones")
    _registrar(registro, f"Dimensiones: A = {filas_a}x{columnas_a}, B = {filas_b}x{columnas_b}")
    registrar_matriz(registro, "A", matriz_a)
    registrar_matriz(registro, "B", matriz_b)
    resultado = [
        [matriz_a[fila][columna] - matriz_b[fila][columna] for columna in range(columnas_a)]
        for fila in range(filas_a)
    ]
    registrar_matriz(registro, "Resultado", resultado)
    return resultado


def multiplicar_matrices(matriz_a: Matriz, matriz_b: Matriz, registro: Optional[Registro] = None) -> Matriz:
    """Multiplica una matriz A(mxn) por una matriz B(nxp)."""
    _registrar(registro, "Producto de matrices: C = A * B")
    _validar_rectangular(matriz_a)
    _validar_rectangular(matriz_b)
    filas_a, columnas_a = dimensiones_matriz(matriz_a)
    filas_b, columnas_b = dimensiones_matriz(matriz_b)
    if columnas_a != filas_b:
        raise ValueError(
            f"Para multiplicar se requiere que columnas de A (= {columnas_a}) coincidan con filas de B (= {filas_b})"
        )
    _registrar(registro, f"Dimensiones: A = {filas_a}x{columnas_a}, B = {filas_b}x{columnas_b}")
    registrar_matriz(registro, "A", matriz_a)
    registrar_matriz(registro, "B", matriz_b)
    resultado: Matriz = [[Fraction(0) for _ in range(columnas_b)] for _ in range(filas_a)]
    for fila in range(filas_a):
        for indice in range(columnas_a):
            valor_a = matriz_a[fila][indice]
            if valor_a == 0:
                continue
            for columna in range(columnas_b):
                resultado[fila][columna] += valor_a * matriz_b[indice][columna]
        if registro is not None and filas_a * columnas_b <= 9 and columnas_a <= 5:
            for columna in range(columnas_b):
                terminos = [
                    f"{formatear_numero(matriz_a[fila][k])}*{formatear_numero(matriz_b[k][columna])}"
                    for k in range(columnas_a)
                    if matriz_a[fila][k] != 0 or matriz_b[k][columna] != 0
                ]
                _registrar(
                    registro,
                    f"C[{fila + 1},{columna + 1}] = " + " + ".join(terminos) + f" = {formatear_numero(resultado[fila][columna])}"
                )
    registrar_matriz(registro, "Resultado", resultado)
    return resultado


def sumar_matrices_lista(matrices: List[Matriz], registro: Optional[Registro] = None) -> Matriz:
    """Suma una lista de matrices del mismo tamano."""
    _registrar(registro, "Suma de varias matrices: C = M1 + M2 + ...")
    if not matrices:
        return []
    _validar_rectangular(matrices[0])
    filas_base, columnas_base = dimensiones_matriz(matrices[0])
    acumulado = [[Fraction(0) for _ in range(columnas_base)] for _ in range(filas_base)]
    for indice, matriz in enumerate(matrices, start=1):
        _validar_rectangular(matriz)
        filas, columnas = dimensiones_matriz(matriz)
        if (filas, columnas) != (filas_base, columnas_base):
            raise ValueError(
                f"Todas las matrices deben tener el mismo tamano; la matriz {indice} es {filas}x{columnas} y se esperaba {filas_base}x{columnas_base}"
            )
        registrar_matriz(registro, f"M{indice}", matriz)
        for fila in range(filas_base):
            for columna in range(columnas_base):
                acumulado[fila][columna] += matriz[fila][columna]
    registrar_matriz(registro, "Resultado", acumulado)
    return acumulado


def restar_matrices_secuencial(matrices: List[Matriz], registro: Optional[Registro] = None) -> Matriz:
    """Resta matrices de forma secuencial: M1 - M2 - M3 - ..."""
    _registrar(registro, "Resta secuencial: R = M1 - M2 - ...")
    if not matrices:
        return []
    _validar_rectangular(matrices[0])
    filas_base, columnas_base = dimensiones_matriz(matrices[0])
    acumulado = [fila[:] for fila in matrices[0]]
    registrar_matriz(registro, "M1", matrices[0])
    for indice, matriz in enumerate(matrices[1:], start=2):
        _validar_rectangular(matriz)
        filas, columnas = dimensiones_matriz(matriz)
        if (filas, columnas) != (filas_base, columnas_base):
            raise ValueError(
                f"Todas las matrices deben tener el mismo tamano; la matriz {indice} es {filas}x{columnas} y se esperaba {filas_base}x{columnas_base}"
            )
        _registrar(registro, f"R := R - M{indice}")
        registrar_matriz(registro, f"M{indice}", matriz)
        for fila in range(filas_base):
            for columna in range(columnas_base):
                acumulado[fila][columna] -= matriz[fila][columna]
    registrar_matriz(registro, "Resultado", acumulado)
    return acumulado


def multiplicar_cadena(matrices: List[Matriz], registro: Optional[Registro] = None) -> Matriz:
    """Multiplica una cadena de matrices: M1 * M2 * ... * Mk."""
    _registrar(registro, "Producto encadenado: R = M1 * M2 * ...")
    if not matrices:
        return []
    acumulado = matrices[0]
    registrar_matriz(registro, "M1", matrices[0])
    for indice, matriz in enumerate(matrices[1:], start=2):
        _registrar(registro, f"Paso {indice - 1}: R := R * M{indice}")
        registrar_matriz(registro, f"M{indice}", matriz)
        acumulado = multiplicar_matrices(acumulado, matriz, registro)
    registrar_matriz(registro, "Resultado", acumulado)
    return acumulado

def transpuesta(matriz: Matriz, registro: Optional[Registro] = None) -> Matriz:
    """Devuelve la transpuesta de una matriz A -> A^T."""
    _registrar(registro, "Transpuesta: T := A^T")
    _validar_rectangular(matriz)
    filas, columnas = dimensiones_matriz(matriz)
    _registrar(registro, f"Dimensiones: A = {filas}x{columnas}")
    registrar_matriz(registro, "A", matriz)
    if filas == 0 or columnas == 0:
        return []
    T: Matriz = [[Fraction(0) for _ in range(filas)] for _ in range(columnas)]
    for i in range(filas):
        for j in range(columnas):
            T[j][i] = matriz[i][j]
    registrar_matriz(registro, "A^T", T)
    return T


def es_multiplicable(A: Matriz, B: Matriz) -> Tuple[bool, str]:
    """Chequeo (ligero) de compatibilidad de dimensiones para A*B."""
    fa, ca = dimensiones_matriz(A)
    fb, cb = dimensiones_matriz(B)
    ok = (ca == fb)
    msg = f"A es {fa}x{ca}, B es {fb}x{cb}. " + ("Compatible." if ok else "Incompatible.")
    return ok, msg


def vector_columna(valores: Sequence[NumeroLike]) -> Matriz:
    """Convierte [a,b,c] en un vector columna 3x1."""
    return [[_a_fraccion(v)] for v in valores]


def vector_fila(valores: Sequence[NumeroLike]) -> Matriz:
    """Convierte [a,b,c] en un vector fila 1x3."""
    return [[_a_fraccion(v) for v in valores]]

def identidad(n: int) -> Matriz:
    return [[Fraction(1) if i == j else Fraction(0) for j in range(n)] for i in range(n)]

def inversa_gauss_jordan(A: Matriz, registro: Optional[Registro] = None, tol: float = 1e-10) -> Matriz:
    """
    Calcula A^{-1} aplicando Gauss–Jordan a [A | I].
    Devuelve la matriz inversa si existe; si no, lanza ValueError.
    """
    _registrar(registro, "Inversa por Gauss–Jordan: [A | I] -> [I | A^{-1}]")
    _validar_rectangular(A)
    n, m = dimensiones_matriz(A)
    if n != m:
        raise ValueError("La inversa solo está definida para matrices cuadradas")

    # Copias de trabajo
    L = copiar_matriz(A)
    R = identidad(n)

    fila_piv = 0
    for col in range(n):
        # Elegir pivote por valor absoluto máximo
        piv = max(range(fila_piv, n), key=lambda r: abs(L[r][col]))
        if abs(L[piv][col]) <= tol:
            _registrar(registro, f"Columna {col+1}: no hay pivote (rango < n)")
            raise ValueError("A no es invertible (rango < n)")

        # Intercambio
        if piv != fila_piv:
            L[fila_piv], L[piv] = L[piv], L[fila_piv]
            R[fila_piv], R[piv] = R[piv], R[fila_piv]
            _registrar(registro, f"Intercambio F{fila_piv+1} <-> F{piv+1}")
            registrar_matriz(registro, "L", L)

        # Normalizar fila pivote
        pivote = L[fila_piv][col]
        if abs(pivote - 1) > tol:
            for c in range(n):
                L[fila_piv][c] /= pivote
                R[fila_piv][c] /= pivote
            _registrar(registro, f"F{fila_piv+1} := F{fila_piv+1} / {formatear_numero(pivote)}")
            registrar_matriz(registro, "L", L)

        # Anular el resto de la columna
        for r in range(n):
            if r == fila_piv:
                continue
            factor = L[r][col]
            if abs(factor) <= tol:
                continue
            for c in range(n):
                L[r][c] -= factor * L[fila_piv][c]
                R[r][c] -= factor * R[fila_piv][c]
            _registrar(registro, f"F{r+1} := F{r+1} - {formatear_numero(factor)} * F{fila_piv+1}")
            registrar_matriz(registro, "L", L)

        fila_piv += 1

    # Limpieza numérica
    for i in range(n):
        for j in range(n):
            if abs(L[i][j]) <= tol:
                L[i][j] = Fraction(0)
            if abs(R[i][j]) <= tol:
                R[i][j] = Fraction(0)

    # Verificar que L == I
    for i in range(n):
        for j in range(n):
            objetivo = Fraction(1) if i == j else Fraction(0)
            if abs(L[i][j] - objetivo) > 1e-8:
                raise ValueError("A no es invertible (no se obtuvo I en la izquierda)")

    registrar_matriz(registro, "A^{-1}", R)
    return R


def caracterizaciones_invertible(A: Matriz, registro: Optional[Registro] = None, tol: float = 1e-10) -> Dict[str, bool]:
    """
    Devuelve un dict con las 12 caracterizaciones clásicas (a–l) del teorema de la matriz invertible.
    """
    _validar_rectangular(A)
    n, m = dimensiones_matriz(A)
    resultado: Dict[str, bool] = {}
    if n != m or n == 0:
        # No cuadrada: todas falsas salvo las que no aplican
        cuadrada = False
        for clave in list("abcdefghijkl"):
            resultado[clave] = False
        return resultado

    cuadrada = True
    # Rango y pivotes con tu rref/rango
    r, Rrref, piv = rango_matriz(A, registro=registro, tol=tol)

    es_inv = (r == n)
    resultado["a"] = es_inv                              # A es invertible
    resultado["b"] = es_inv                              # A ~ I_n por filas
    resultado["c"] = (len(piv) == n)                     # n posiciones pivote
    resultado["d"] = es_inv                              # Ax = 0 solo trivial
    resultado["e"] = es_inv                              # columnas LI
    resultado["f"] = es_inv                              # x ↦ Ax es uno-a-uno
    resultado["g"] = es_inv                              # Ax = b tiene sol. ∀ b
    resultado["h"] = es_inv                              # columnas generan R^n
    resultado["i"] = es_inv                              # x ↦ Ax es sobre R^n
    resultado["j"] = es_inv                              # ∃ C con CA = I
    resultado["k"] = es_inv                              # ∃ D con AD = I
    # (A^T) invertible ↔ A invertible
    AT = transpuesta(A)
    rT, _, _ = rango_matriz(AT, tol=tol)
    resultado["l"] = (rT == n)

    return resultado


__all__ = [
    "Numero",
    "Matriz",
    "Registro",
    "RegistroEntrada",
    "formatear_numero",
    "registrar_matriz",
    "sumar_matrices_dos",
    "restar_matrices_dos",
    "multiplicar_matrices",
    "sumar_matrices_lista",
    "restar_matrices_secuencial",
    "multiplicar_cadena",
    "rref",
    "rango_matriz",
    "transpuesta",
    "es_multiplicable",
    "vector_columna",
    "vector_fila",
    "identidad",
    "inversa_gauss_jordan",
    "caracterizaciones_invertible",
    "determinante",
    "regla_cramer",
]
