import tkinter as tk
from tkinter import ttk, messagebox
import math
from fractions import Fraction
from typing import List, Optional, Union

# Constantes para superíndices y subíndices (Estética)
SUPERSCRIPT_MAP = {"0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹", "+": "⁺", "-": "⁻", "(": "⁽", ")": "⁾"}
SUBSCRIPT_MAP = {"0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉", "+": "₊", "-": "₋", "(": "₍", ")": "₎"}

def to_superscript(text: str) -> str:
    return "".join(SUPERSCRIPT_MAP.get(char, char) for char in text)

def format_fraction_unicode(valor: Fraction) -> str:
    if valor.denominator == 1:
        return str(valor.numerator)
    return f"{valor.numerator}⁄{valor.denominator}"

class SimpleCalculator(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.pack(fill=tk.BOTH, expand=True)

        # Variables de estado
        self.display_var = tk.StringVar(value="0")
        self.expression_var = tk.StringVar(value="")
        self.current_entry = ""
        self.expression_tokens: List[str] = []
        self.last_was_result = False

        self._create_ui()
        self._bind_keys()

    def _create_ui(self):
        # Pantalla
        display_frame = tk.Frame(self, bg="#222", pady=10, padx=10)
        display_frame.pack(fill=tk.X)

        lbl_historial = tk.Label(display_frame, textvariable=self.expression_var, 
                               anchor="e", bg="#222", fg="#888", font=("Segoe UI", 10))
        lbl_historial.pack(fill=tk.X)

        lbl_display = tk.Label(display_frame, textvariable=self.display_var, 
                               anchor="e", bg="#222", fg="white", font=("Segoe UI", 24, "bold"))
        lbl_display.pack(fill=tk.X)

        # Botones
        buttons_frame = tk.Frame(self, bg="#f3f3f3", pady=5)
        buttons_frame.pack(fill=tk.BOTH, expand=True)

        # Configuración de la rejilla (Grid)
        for i in range(5): buttons_frame.grid_columnconfigure(i, weight=1)
        for i in range(6): buttons_frame.grid_rowconfigure(i, weight=1)

        # Layout de botones
        buttons = [
            ('C', 0, 0), ('(', 0, 1), (')', 0, 2), ('⌫', 0, 3), ('÷', 0, 4),
            ('7', 1, 0), ('8', 1, 1), ('9', 1, 2), ('×', 1, 3), ('√', 1, 4),
            ('4', 2, 0), ('5', 2, 1), ('6', 2, 2), ('-', 2, 3), ('x²', 2, 4),
            ('1', 3, 0), ('2', 3, 1), ('3', 3, 2), ('+', 3, 3), ('1/x', 3, 4),
            ('±', 4, 0), ('0', 4, 1), ('.', 4, 2), ('=', 4, 3, 2) # El igual ocupa 2 columnas
        ]

        for btn_info in buttons:
            text = btn_info[0]
            row = btn_info[1]
            col = btn_info[2]
            colspan = btn_info[3] if len(btn_info) > 3 else 1
            
            # Estilos según tipo de botón
            bg_color = "white"
            fg_color = "#333"
            if text in ('C', '⌫'): bg_color, fg_color = "#ffcccc", "#d00000"
            elif text == '=': bg_color, fg_color = "#007acc", "white"
            elif text in ('÷', '×', '-', '+'): bg_color, fg_color = "#e6e6e6", "#005a9e"

            btn = tk.Button(buttons_frame, text=text, font=("Segoe UI", 14),
                            bg=bg_color, fg=fg_color, bd=0, cursor="hand2",
                            command=lambda t=text: self._on_button_click(t))
            btn.grid(row=row, column=col, columnspan=colspan, sticky="nsew", padx=2, pady=2)

    def _bind_keys(self):
        # Soporte para teclado físico
        key_map = {
            '<Return>': '=', '<KP_Enter>': '=', '<BackSpace>': '⌫', '<Escape>': 'C',
            '+': '+', '-': '-', '*': '×', '/': '÷', '.': '.',
            '(': '(', ')': ')'
        }
        for i in range(10): key_map[str(i)] = str(i)
        
        # Binds globales en la ventana raíz
        top = self.winfo_toplevel()
        for key, val in key_map.items():
            top.bind(key, lambda e, v=val: self._on_button_click(v))

    def _on_button_click(self, char):
        if char.isdigit(): self._add_digit(char)
        elif char == '.': self._add_decimal()
        elif char in ('+', '-', '×', '÷'): self._set_operator(char)
        elif char == '=': self._calculate()
        elif char == 'C': self._clear()
        elif char == '⌫': self._backspace()
        elif char == '±': self._toggle_sign()
        elif char == '√': self._sqrt()
        elif char == 'x²': self._square()
        elif char == '1/x': self._reciprocal()
        elif char in ('(', ')'): self._parenthesis(char)

    # --- Lógica de Calculadora ---
    def _update_display(self):
        self.display_var.set(self.current_entry if self.current_entry else "0")

    def _add_digit(self, digit):
        if self.last_was_result:
            self.current_entry = ""
            self.last_was_result = False
        if self.current_entry == "0": self.current_entry = digit
        else: self.current_entry += digit
        self._update_display()

    def _add_decimal(self):
        if self.last_was_result:
            self.current_entry = "0"
            self.last_was_result = False
        if "." not in self.current_entry:
            self.current_entry += "." if self.current_entry else "0."
        self._update_display()

    def _clear(self):
        self.current_entry = ""
        self.expression_tokens = []
        self.expression_var.set("")
        self._update_display()

    def _backspace(self):
        if self.last_was_result: return
        self.current_entry = self.current_entry[:-1]
        self._update_display()

    def _set_operator(self, op):
        if self.current_entry:
            self.expression_tokens.append(self.current_entry)
            self.current_entry = ""
        elif self.last_was_result and self.display_var.get() != "Error":
             # Usar resultado anterior si se presiona operador directo
             self.expression_tokens.append(self.display_var.get())
             self.last_was_result = False
        
        # Reemplazar operador si se presiona otro seguido
        if self.expression_tokens and self.expression_tokens[-1] in ('+', '-', '*', '/'):
            self.expression_tokens.pop()
        
        op_map = {'×': '*', '÷': '/'}
        self.expression_tokens.append(op_map.get(op, op))
        
        # Actualizar visor superior
        expr_str = " ".join(self.expression_tokens).replace('*', '×').replace('/', '÷')
        self.expression_var.set(expr_str)

    def _calculate(self):
        if self.current_entry:
            self.expression_tokens.append(self.current_entry)
        
        if not self.expression_tokens: return

        try:
            expr = "".join(self.expression_tokens)
            # Evaluar con Fractions para precisión
            # Reemplazamos números decimales por Fractions strings
            # Nota: eval es peligroso en general, pero aquí controlamos los tokens
            # Una implementación robusta usaría un parser propio, 
            # pero para este demo usamos eval con cuidado.
            
            # Truco simple: Convertir todo a float para visualización rápida
            # O usar Fractions si quieres precisión exacta
            resultado = eval(expr) 
            
            # Formatear
            if isinstance(resultado, float):
                if resultado.is_integer(): resultado = int(resultado)
                else: resultado = round(resultado, 10)
            
            self.current_entry = str(resultado)
            self.display_var.set(self.current_entry)
            self.expression_var.set(self.expression_var.get() + " =")
            self.expression_tokens = []
            self.last_was_result = True

        except Exception:
            self.display_var.set("Error")
            self.current_entry = ""
            self.expression_tokens = []

    def _toggle_sign(self):
        if self.current_entry:
            if self.current_entry.startswith("-"): self.current_entry = self.current_entry[1:]
            else: self.current_entry = "-" + self.current_entry
            self._update_display()

    def _sqrt(self):
        try:
            val = float(self.current_entry or self.display_var.get())
            if val < 0: raise ValueError
            res = math.sqrt(val)
            self.current_entry = str(int(res) if res.is_integer() else round(res, 8))
            self.last_was_result = True
            self._update_display()
        except: self.display_var.set("Error")

    def _square(self):
        try:
            val = float(self.current_entry or self.display_var.get())
            res = val ** 2
            self.current_entry = str(int(res) if res.is_integer() else round(res, 8))
            self.last_was_result = True
            self._update_display()
        except: self.display_var.set("Error")

    def _reciprocal(self):
        try:
            val = float(self.current_entry or self.display_var.get())
            if val == 0: raise ValueError
            res = 1 / val
            self.current_entry = str(int(res) if res.is_integer() else round(res, 8))
            self.last_was_result = True
            self._update_display()
        except: self.display_var.set("Error")
        
    def _parenthesis(self, char):
        if self.current_entry:
            self.expression_tokens.append(self.current_entry)
            self.current_entry = ""
        self.expression_tokens.append(char)
        self.expression_var.set(" ".join(self.expression_tokens))