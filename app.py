import tkinter as tk
from tkinter import ttk
from ui_components import MenuLateral, DashboardCard, COLOR_FONDO_PRINCIPAL
from simple_calculator import SimpleCalculator
from views_matrix import VentanaCalculadoraUniversal, VentanaSistemas, VentanaVectores
from views_numerical import VistaNewton, VistaSecante, VentanaBiseccion, VentanaReglaFalsa

class Aplicacion(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Suite Matemática Pro")
        self.geometry("1200x750")
        self.minsize(1000, 600)
        self.config(bg=COLOR_FONDO_PRINCIPAL) 

        self.MODULOS = {
            "calculadora": {
                "titulo": "Calculadora", "subtitulo": "Científica Básica", "icono": "🧮", "color": "#E67E22",
                "vistas": [("calc_basic", "Básica", SimpleCalculator)]
            },
            "matrices": {
                "titulo": "Álgebra Lineal", "subtitulo": "Espacio de Trabajo", "icono": "📊", "color": "#8E44AD",
                "vistas": [
                    ("calc_mat", "Calculadora Matricial", VentanaCalculadoraUniversal),
                    ("sistemas", "Sistemas (Ax=b)", VentanaSistemas),
                    ("vectores", "Espacios Vectoriales", VentanaVectores),
                ]
            },
            "numericos": {
                "titulo": "Métodos Numéricos", "subtitulo": "Raíces", "icono": "📈", "color": "#27AE60",
                "vistas": [
                    ("newton", "Newton-Raphson", VistaNewton),
                    ("secante", "Secante", VistaSecante),
                    ("biseccion", "Bisección", VentanaBiseccion),
                    ("falsa", "Regla Falsa", VentanaReglaFalsa),
                ]
            }
        }
        self.main_container = tk.Frame(self, bg=COLOR_FONDO_PRINCIPAL)
        self.main_container.pack(fill="both", expand=True)
        self._mostrar_dashboard()

    def _mostrar_dashboard(self):
        for w in self.main_container.winfo_children(): w.destroy()
        
        header = tk.Frame(self.main_container, bg="white", height=80, padx=50)
        header.pack(fill="x")
        tk.Label(header, text="Math Suite", font=("Segoe UI", 22, "bold"), fg="#2C3E50", bg="white").pack(side="left", pady=20)
        
        grid = tk.Frame(self.main_container, bg=COLOR_FONDO_PRINCIPAL)
        grid.place(relx=0.5, rely=0.5, anchor="center")

        for i, (key, data) in enumerate(self.MODULOS.items()):
            DashboardCard(grid, data["titulo"], data["subtitulo"], data["icono"], data["color"], lambda k=key: self._load_mod(k)).grid(row=0, column=i, padx=20)

    def _load_mod(self, key):
        self.active_mod = key
        data = self.MODULOS[key]
        for w in self.main_container.winfo_children(): w.destroy()
        
        sidebar = MenuLateral(self.main_container, data["titulo"], [(v[1], v[0]) for v in data["vistas"]], self._change_tool, self._mostrar_dashboard, data["color"])
        sidebar.pack(side=tk.LEFT, fill="y")
        self.sidebar = sidebar
        
        self.work_area = tk.Frame(self.main_container, bg="white")
        self.work_area.pack(side=tk.RIGHT, fill="both", expand=True)
        self._change_tool(data["vistas"][0][0])

    def _change_tool(self, key):
        for w in self.work_area.winfo_children(): w.destroy()
        target = next(v for v in self.MODULOS[self.active_mod]["vistas"] if v[0] == key)
        
        tk.Label(self.work_area, text=target[1], font=("Segoe UI", 18), bg="white", fg="#333").pack(anchor="sw", padx=30, pady=10)
        ttk.Separator(self.work_area).pack(fill="x")
        
        container = tk.Frame(self.work_area, bg="white", padx=20, pady=20)
        container.pack(fill="both", expand=True)
        try: target[2](container)
        except Exception as e: tk.Label(container, text=f"Error: {e}", fg="red").pack()
        self.sidebar.marcar_seleccion(key)

if __name__ == "__main__":
    app = Aplicacion()
    app.mainloop()