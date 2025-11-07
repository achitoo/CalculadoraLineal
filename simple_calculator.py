import math
import tkinter as tk
from fractions import Fraction
from tkinter import ttk, messagebox
from typing import List, Optional, Sequence, Tuple, Union

SUPERSCRIPT_MAP = {
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "+": "⁺",
    "-": "⁻",
    "(": "⁽",
    ")": "⁾",
}

SUBSCRIPT_MAP = {
    "0": "₀",
    "1": "₁",
    "2": "₂",
    "3": "₃",
    "4": "₄",
    "5": "₅",
    "6": "₆",
    "7": "₇",
    "8": "₈",
    "9": "₉",
    "+": "₊",
    "-": "₋",
    "(": "₍",
    ")": "₎",
}

FRACTION_SLASH = "\u2044"
NumberToken = Union[Fraction, float]


def to_superscript(text: str) -> str:
    return "".join(SUPERSCRIPT_MAP.get(char, char) for char in text)


def to_subscript(text: str) -> str:
    return "".join(SUBSCRIPT_MAP.get(char, char) for char in text)


def format_fraction_unicode(valor: Fraction) -> str:
    if valor.denominator == 1:
        return str(valor.numerator)
    signo = "-" if valor < 0 else ""
    numerador = to_superscript(str(abs(valor.numerator)))
    denominador = to_subscript(str(valor.denominator))
    return f"{signo}{numerador}{FRACTION_SLASH}{denominador}"


def format_float(value: float, precision: int = 10) -> str:
    texto = f"{value:.{precision}g}"
    if "e" not in texto and "E" not in texto and "." in texto:
        texto = texto.rstrip("0").rstrip(".")
    return texto


class SimpleCalculator(ttk.Frame):
    """Calculadora básica con soporte para fracciones, exponentes y raíces."""

    def __init__(self, maestro: tk.Misc) -> None:
        super().__init__(maestro, padding=12)

        self.display_var = tk.StringVar(value="0")
        self.expression_var = tk.StringVar(value="")
        self.current_entry = ""
        self.accumulator: Optional[NumberToken] = None
        self.pending_operator: Optional[str] = None
        self.last_operand: Optional[NumberToken] = None
        self.expression_tokens: List[Tuple[str, Union[NumberToken, str]]] = []
        self.last_was_result = False

        self._build_ui()
        self._update_display()

    # ------------------------------------------------------------------ UI --
    def _build_ui(self) -> None:
        display_frame = tk.Frame(self, bg="#111111", bd=1, relief="sunken")
        display_frame.pack(fill="x", pady=(0, 8))

        self.expression_label = tk.Label(
            display_frame,
            textvariable=self.expression_var,
            anchor="e",
            font=("Segoe UI", 12),
            bg="#111111",
            fg="#bbbbbb",
            padx=4,
            pady=2,
        )
        self.expression_label.pack(fill="x")

        self.display = tk.Label(
            display_frame,
            textvariable=self.display_var,
            anchor="e",
            font=("Segoe UI", 26),
            bg="#111111",
            fg="#f5f5f5",
            padx=4,
            pady=6,
        )
        self.display.pack(fill="x")

        buttons_frame = ttk.Frame(self)
        buttons_frame.pack(fill="both", expand=True)

        layout: Sequence[Sequence[Tuple[str, str]]] = [
            (("CE", "clear_entry"), ("C", "clear_all"), ("DEL", "backspace"), ("/", "op")),
            (("√", "sqrt"), ("^", "op"), ("a/b", "slash"), ("*", "op")),
            (("7", "digit"), ("8", "digit"), ("9", "digit"), ("-", "op")),
            (("4", "digit"), ("5", "digit"), ("6", "digit"), ("+", "op")),
            (("1", "digit"), ("2", "digit"), ("3", "digit"), ("=", "equals")),
            (("+/-", "sign"), ("0", "digit"), (".", "decimal"), ("", "")),
        ]

        for fila, fila_def in enumerate(layout):
            for columna, (texto, accion) in enumerate(fila_def):
                if not texto:
                    continue
                boton = ttk.Button(buttons_frame, text=texto, width=6)
                boton.grid(row=fila, column=columna, padx=4, pady=4, sticky="nsew")
                if accion == "digit":
                    boton.configure(command=lambda t=texto: self._append_digit(t))
                elif accion == "decimal":
                    boton.configure(command=self._insert_decimal)
                elif accion == "slash":
                    boton.configure(command=self._insert_slash)
                elif accion == "op":
                    boton.configure(command=lambda t=texto: self._on_operator("^" if t == "^" else t))
                elif accion == "equals":
                    boton.configure(command=self._on_equals)
                elif accion == "sqrt":
                    boton.configure(command=self._sqrt)
                elif accion == "sign":
                    boton.configure(command=self._toggle_sign)
                elif accion == "clear_entry":
                    boton.configure(command=self._clear_entry)
                elif accion == "clear_all":
                    boton.configure(command=self._clear_all)
                elif accion == "backspace":
                    boton.configure(command=self._backspace)

        for i in range(4):
            buttons_frame.columnconfigure(i, weight=1)
        for i in range(len(layout)):
            buttons_frame.rowconfigure(i, weight=1)

    # ------------------------------------------------------------ helpers --
    def _ensure_fraction(self, value: NumberToken) -> Fraction:
        if isinstance(value, Fraction):
            return value
        return Fraction(str(value)).limit_denominator(10**6)

    def _format_number_display(self, value: NumberToken) -> str:
        if isinstance(value, Fraction):
            return format_fraction_unicode(value)
        return format_float(float(value))

    def _format_entry_display(self, entry: str) -> str:
        if not entry:
            return ""
        try:
            valor = self._parse_entry(entry)
        except ValueError:
            return entry
        return self._format_number_display(valor)

    def _format_exponent_display(self, valor: NumberToken) -> str:
        if isinstance(valor, Fraction):
            if valor.denominator == 1:
                return to_superscript(str(valor.numerator))
            return f"^{format_fraction_unicode(valor)}"
        if isinstance(valor, float) and valor.is_integer():
            return to_superscript(str(int(valor)))
        return f"^{format_float(float(valor))}"

    def _map_operator_symbol(self, operador: str) -> str:
        return {"*": "×", "/": "÷"}.get(operador, operador)

    def _append_number_token(self, valor: NumberToken) -> None:
        if not self.expression_tokens or self.expression_tokens[-1][0] == "operator":
            self.expression_tokens.append(("number", valor))
        else:
            self.expression_tokens[-1] = ("number", valor)

    def _append_operator_token(self, operador: str) -> None:
        if not self.expression_tokens:
            if self.accumulator is None:
                return
            self.expression_tokens.append(("number", self.accumulator))
        if self.expression_tokens[-1][0] == "operator":
            self.expression_tokens[-1] = ("operator", operador)
        else:
            self.expression_tokens.append(("operator", operador))

    def _build_expression_string(self, include_current: bool = True) -> str:
        tokens: List[str] = []
        current_display = self._format_entry_display(self.current_entry) if include_current and self.current_entry else ""
        current_used = False
        i = 0
        while i < len(self.expression_tokens):
            tipo, valor = self.expression_tokens[i]
            if tipo == "number":
                tokens.append(self._format_number_display(valor))  # type: ignore[arg-type]
            elif tipo == "operator":
                operador = valor  # type: ignore[assignment]
                if operador == "^":
                    base = tokens.pop() if tokens else ""
                    exponent_text = "^"
                    if i + 1 < len(self.expression_tokens) and self.expression_tokens[i + 1][0] == "number":
                        exponent_text = self._format_exponent_display(self.expression_tokens[i + 1][1])  # type: ignore[index]
                        i += 1
                    elif current_display and not current_used:
                        try:
                            exponent_value = self._parse_entry(self.current_entry)
                            exponent_text = self._format_exponent_display(exponent_value)
                            current_used = True
                        except ValueError:
                            exponent_text = "^"
                    tokens.append(f"{base}{exponent_text}")
                else:
                    tokens.append(self._map_operator_symbol(operador))
            i += 1
        if current_display and not current_used:
            tokens.append(current_display)
        return " ".join(tokens).strip()

    def _update_display(self) -> None:
        if self.current_entry:
            self.display_var.set(self._format_entry_display(self.current_entry))
        elif self.accumulator is not None:
            self.display_var.set(self._format_number_display(self.accumulator))
        else:
            self.display_var.set("0")
        self.expression_var.set(self._build_expression_string())

    # ----------------------------------------------------------- acciones --
    def _append_digit(self, digito: str) -> None:
        if self.display_var.get() == "Error":
            self._clear_all()
        if self.last_was_result:
            self.current_entry = ""
            self.expression_tokens.clear()
            self.accumulator = None
            self.pending_operator = None
            self.expression_var.set("")
            self.last_was_result = False
        if self.current_entry == "0":
            self.current_entry = digito
        else:
            self.current_entry += digito
        self._update_display()

    def _insert_decimal(self) -> None:
        if self.display_var.get() == "Error":
            self._clear_all()
        if self.last_was_result:
            self.current_entry = ""
            self.expression_tokens.clear()
            self.accumulator = None
            self.pending_operator = None
            self.expression_var.set("")
            self.last_was_result = False
        if "/" in self.current_entry or "." in self.current_entry:
            return
        if not self.current_entry:
            self.current_entry = "0."
        else:
            self.current_entry += "."
        self._update_display()

    def _insert_slash(self) -> None:
        if self.display_var.get() == "Error":
            self._clear_all()
        if self.last_was_result:
            self.current_entry = ""
            self.expression_tokens.clear()
            self.accumulator = None
            self.pending_operator = None
            self.expression_var.set("")
            self.last_was_result = False
        if "/" in self.current_entry:
            return
        if not self.current_entry:
            self.current_entry = "0/"
        else:
            self.current_entry += "/"
        self._update_display()

    def _toggle_sign(self) -> None:
        objetivo = self.current_entry or self.display_var.get()
        if objetivo in {"", "0", "Error"}:
            return
        try:
            valor = self._parse_entry(objetivo)
        except ValueError:
            self._set_error("Entrada invalida")
            return
        valor = -valor
        self.current_entry = str(valor)
        self.last_was_result = False
        self._update_display()

    def _clear_entry(self) -> None:
        self.current_entry = ""
        self.last_was_result = False
        self._update_display()

    def _clear_all(self) -> None:
        self.current_entry = ""
        self.accumulator = None
        self.pending_operator = None
        self.last_operand = None
        self.expression_tokens.clear()
        self.expression_var.set("")
        self.last_was_result = False
        self._update_display()

    def _backspace(self) -> None:
        if self.display_var.get() == "Error":
            self._clear_all()
            return
        if not self.current_entry:
            self.current_entry = self.display_var.get()
        if self.current_entry in {"", "0"}:
            return
        self.current_entry = self.current_entry[:-1]
        self._update_display()

    def _on_operator(self, operador: str) -> None:
        if self.display_var.get() == "Error":
            self._clear_all()
        if self.current_entry:
            try:
                valor = self._parse_entry(self.current_entry)
            except ValueError:
                self._set_error("Entrada invalida")
                return
            self._append_number_token(valor)
            if self.accumulator is None or self.pending_operator is None:
                self.accumulator = valor
            else:
                try:
                    self.accumulator = self._operate(self.accumulator, valor, self.pending_operator)
                except ZeroDivisionError:
                    self._set_error("Division por cero")
                    return
            self.last_operand = valor
            self.current_entry = ""
        elif self.accumulator is None:
            return
        self._append_operator_token(operador)
        self.pending_operator = operador
        self.last_was_result = False
        self._update_display()

    def _operate(self, a: NumberToken, b: NumberToken, operador: str) -> NumberToken:
        if operador == "^":
            base = self._ensure_fraction(a)
            exponente = self._ensure_fraction(b)
            if exponente.denominator == 1:
                return base ** exponente.numerator
            return float(base) ** float(exponente)
        frac_a = self._ensure_fraction(a)
        frac_b = self._ensure_fraction(b)
        if operador == "+":
            return frac_a + frac_b
        if operador == "-":
            return frac_a - frac_b
        if operador == "*":
            return frac_a * frac_b
        if operador == "/":
            if frac_b == 0:
                raise ZeroDivisionError
            return frac_a / frac_b
        raise ValueError(f"Operador desconocido: {operador}")

    def _on_equals(self) -> None:
        if self.display_var.get() == "Error":
            self._clear_all()
            return
        if self.pending_operator is None:
            if self.current_entry:
                try:
                    valor = self._parse_entry(self.current_entry)
                except ValueError:
                    self._set_error("Entrada invalida")
                    return
                self.accumulator = valor
                self.expression_tokens = [("number", valor)]
                self._set_result(valor, self._build_expression_string(include_current=False))
            return

        if self.current_entry:
            try:
                valor = self._parse_entry(self.current_entry)
            except ValueError:
                self._set_error("Entrada invalida")
                return
            self._append_number_token(valor)
            self.last_operand = valor
        else:
            if self.last_operand is not None:
                self._append_number_token(self.last_operand)
                valor = self.last_operand
            elif self.expression_tokens and self.expression_tokens[-1][0] == "number":
                valor = self.expression_tokens[-1][1]  # type: ignore[index]
            else:
                valor = self.accumulator if self.accumulator is not None else Fraction(0)
                self._append_number_token(valor)
                self.last_operand = valor

        try:
            resultado = self._operate(self.accumulator or Fraction(0), valor, self.pending_operator)
        except ZeroDivisionError:
            self._set_error("Division por cero")
            return
        expression_text = self._build_expression_string(include_current=False)
        self._set_result(resultado, expression_text)

    def _set_result(self, resultado: NumberToken, expression_text: Optional[str] = None) -> None:
        self.accumulator = resultado
        display = self._format_number_display(resultado)
        if expression_text:
            self.expression_var.set(f"{expression_text} = {display}")
        else:
            self.expression_var.set(display)
        self.display_var.set(display)
        self.pending_operator = None
        self.current_entry = ""
        self.expression_tokens.clear()
        self.last_was_result = True

    def _sqrt(self) -> None:
        objetivo = self.current_entry or self.display_var.get()
        try:
            valor = self._parse_entry(objetivo)
        except ValueError:
            self._set_error("Entrada invalida")
            return
        if valor < 0:
            self._set_error("Raiz de numero negativo")
            return
        raiz = math.sqrt(float(valor))
        if raiz.is_integer():
            self.current_entry = str(int(raiz))
        else:
            self.current_entry = str(Fraction(raiz).limit_denominator(10**6))
        self.last_was_result = False
        self._update_display()

    # ------------------------------------------------------------- extras --
    def _parse_entry(self, texto: str) -> Fraction:
        texto = texto.strip()
        if not texto or texto in {".", "-", "-.", "Error"}:
            raise ValueError("Entrada vacia")
        if texto.endswith("."):
            texto = texto[:-1]
        return Fraction(texto)

    def _set_error(self, mensaje: str) -> None:
        self.display_var.set("Error")
        self.expression_var.set("")
        self.current_entry = ""
        self.accumulator = None
        self.pending_operator = None
        self.last_operand = None
        self.expression_tokens.clear()
        self.last_was_result = True
        if mensaje:
            messagebox.showerror("Error", mensaje, parent=self.winfo_toplevel())


__all__ = ["SimpleCalculator", "format_fraction_unicode", "to_superscript", "to_subscript"]
