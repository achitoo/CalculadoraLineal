import tkinter as tk

# --- PALETA DE COLORES TIPO "WEB MODERNA" ---
COLOR_BARRA_LATERAL = "#2C3E50" # Azul oscuro elegante
COLOR_FONDO_PRINCIPAL = "#ECF0F1" # Gris muy claro (casi blanco)
COLOR_TEXTO_SIDEBAR = "#BDC3C7"
COLOR_HOVER_SIDEBAR = "#34495E"
COLOR_ACCENT = "#3498DB" # Azul brillante para destacados

class ModernButton(tk.Button):
    """Botón plano con efecto hover suave."""
    def __init__(self, parent, text, command, icon=None, bg=COLOR_BARRA_LATERAL, fg=COLOR_TEXTO_SIDEBAR, hover_bg=COLOR_HOVER_SIDEBAR):
        super().__init__(parent, text=f"  {text}", command=command, 
                         bg=bg, fg=fg, bd=0, 
                         activebackground=hover_bg, activeforeground="white",
                         font=("Segoe UI", 11), anchor="w", padx=20, pady=10, cursor="hand2")
        self.bg_normal = bg
        self.bg_hover = hover_bg
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        self.config(bg=self.bg_hover, fg="white")

    def on_leave(self, e):
        self.config(bg=self.bg_normal, fg=COLOR_TEXTO_SIDEBAR)


class DashboardCard(tk.Frame):
    """Tarjeta moderna con sombra simulada (Borde suave)."""
    def __init__(self, parent, titulo, subtitulo, icono, color_accent, command):
        # Marco exterior (Simula borde/sombra)
        super().__init__(parent, bg="#DCDCDC", bd=0, padx=1, pady=1, cursor="hand2")
        self.command = command
        
        # Contenido interno blanco
        self.inner = tk.Frame(self, bg="white", padx=20, pady=20)
        self.inner.pack(fill=tk.BOTH, expand=True)
        
        # Banda de color superior
        self.bar = tk.Frame(self.inner, bg=color_accent, height=4)
        self.bar.pack(fill=tk.X, side=tk.TOP, pady=(0, 15))
        
        # Icono grande
        self.lbl_icon = tk.Label(self.inner, text=icono, font=("Segoe UI Emoji", 36), bg="white", fg="#555")
        self.lbl_icon.pack(anchor="center", pady=(0, 10))
        
        # Título
        self.lbl_title = tk.Label(self.inner, text=titulo, font=("Segoe UI", 14, "bold"), bg="white", fg="#333")
        self.lbl_title.pack(anchor="center")
        
        # Subtítulo
        self.lbl_sub = tk.Label(self.inner, text=subtitulo, font=("Segoe UI", 9), bg="white", fg="#888", wraplength=180, justify="center")
        self.lbl_sub.pack(anchor="center", pady=(5, 0))
        
        # Bindings para click
        for w in [self, self.inner, self.lbl_icon, self.lbl_title, self.lbl_sub, self.bar]:
            w.bind("<Button-1>", lambda e: command())
            w.bind("<Enter>", self.on_enter)
            w.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        self.config(bg="#AAAAAA") # Oscurecer borde para efecto "Highlight"

    def on_leave(self, e):
        self.config(bg="#DCDCDC")


class MenuLateral(tk.Frame):
    """Barra lateral completa estilo Dashboard Web."""
    def __init__(self, master, titulo_modulo, opciones, on_select, on_back, color_tema, **kwargs):
        super().__init__(master, bg=COLOR_BARRA_LATERAL, **kwargs)
        self.on_select = on_select
        self.botones = {}
        
        # --- HEADER DEL SIDEBAR (Logo/Título) ---
        header = tk.Frame(self, bg=color_tema, height=80, padx=20)
        header.pack(fill="x")
        header.pack_propagate(False) # Forzar altura fija
        
        # Botón Volver sutil
        btn_back = tk.Label(header, text="⬅ Inicio", font=("Segoe UI", 9), 
                            bg=color_tema, fg="white", cursor="hand2")
        btn_back.pack(anchor="nw", pady=(10, 0))
        btn_back.bind("<Button-1>", lambda e: on_back())
        
        # Título grande
        tk.Label(header, text=titulo_modulo, font=("Segoe UI", 16, "bold"), 
                 bg=color_tema, fg="white").pack(anchor="sw", pady=(0, 10))

        # --- LISTA DE OPCIONES ---
        # Contenedor para botones
        self.nav_frame = tk.Frame(self, bg=COLOR_BARRA_LATERAL, pady=10)
        self.nav_frame.pack(fill="both", expand=True)

        for nombre, clave in opciones:
            btn = ModernButton(
                self.nav_frame,
                text=nombre,
                command=lambda c=clave: self.on_select(c),
                bg=COLOR_BARRA_LATERAL,
                hover_bg=COLOR_HOVER_SIDEBAR
            )
            btn.pack(fill="x", pady=1)
            self.botones[clave] = btn
            
        # --- FOOTER (Decorativo) ---
        lbl_footer = tk.Label(self, text="v1.0.0", bg=COLOR_BARRA_LATERAL, fg="#7F8C8D", font=("Segoe UI", 8))
        lbl_footer.pack(side=tk.BOTTOM, pady=10)

    def marcar_seleccion(self, clave_seleccionada):
        """Marca visualmente qué herramienta está activa."""
        for clave, btn in self.botones.items():
            if clave == clave_seleccionada:
                # Estilo "Activo": Borde izquierdo de color y fondo más claro
                btn.config(bg="#34495E", fg="white", font=("Segoe UI", 11, "bold"))
                # Simulamos borde izquierdo con un frame (truco visual)
                # En Tkinter puro es difícil, así que usamos cambio de color
            else:
                btn.config(bg=COLOR_BARRA_LATERAL, fg=COLOR_TEXTO_SIDEBAR, font=("Segoe UI", 11))