"""
TORO - CÁLCULO TENSORIAL EN SUPERFICIES CURVAS
Gradiente, Diferencial, Tensor Métrico y Derivada Direccional
Superficie: Toro (dona) embebido en R3
"""

import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

DEMETER_COLORSCALE = [
    [0.0, '#1f382a'],
    [0.5, '#7c8c4c'],
    [1.0, '#f2d58e']
]
DEMETER_CMAP = LinearSegmentedColormap.from_list('demeter', ['#1f382a', '#7c8c4c', '#f2d58e'])

# ============================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================

st.set_page_config(
    page_title="Toro - Cálculo Tensorial",
    page_icon="🍩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# ESTILO CSS PERSONALIZADO
# ============================================

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0b10 0%, #1a0a2e 100%);
    }
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #8B0000, #D2691E, #FFD700);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: #FFD700;
        font-family: 'Times New Roman', serif;
        font-size: 2.5rem;
        margin: 0;
    }
    .main-header p {
        color: #FFF8DC;
        font-style: italic;
        margin: 0;
    }
    .card {
        background: rgba(139, 0, 0, 0.3);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #FFD700;
    }
    .formula {
        font-family: 'Courier New', monospace;
        background: #0a0b10;
        padding: 0.5rem;
        border-radius: 5px;
        color: #FFD700;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# ENCABEZADO
# ============================================

st.markdown("""
<div class="main-header">
    <h1>🍩 TORO - CÁLCULO TENSORIAL 🍩</h1>
    <p>Superficie Curva | Gradiente · Diferencial · Tensor Métrico · Derivada Direccional</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# BARRA LATERAL - PARÁMETROS DEL TORO
# ============================================

with st.sidebar:
    st.markdown("## 🍩 Geometría del Toro")
    
    col1, col2 = st.columns(2)
    with col1:
        R = st.number_input("Radio mayor (R)", value=3.0, min_value=1.0, max_value=5.0, step=0.5)
    with col2:
        r = st.number_input("Radio menor (r)", value=1.0, min_value=0.3, max_value=2.0, step=0.1)
    
    st.markdown("---")
    st.markdown("## 🔮 Función Potencial f(θ, φ)")
    
    ejemplos = {
        "cos(θ)·cos(φ)": "cos(theta)*cos(phi)",
        "sin(θ)·sin(φ)": "sin(theta)*sin(phi)",
        "cos(2θ) + sin(2φ)": "cos(2*theta) + sin(2*phi)",
        "θ + φ": "theta + phi",
        "exp(-(θ-π)²) * cos(φ)": "exp(-(theta-3.1416)**2)*cos(phi)"
    }
    
    funcion_ejemplo = st.selectbox("📌 Ejemplos:", list(ejemplos.keys()))
    funcion_str = st.text_input("o escribe f(θ, φ):", 
                                 value=ejemplos[funcion_ejemplo],
                                 help="Usa: theta, phi, sin, cos, exp, ** para potencias")
    
    st.markdown("---")
    st.markdown("## 📍 Punto de Evaluación")
    
    col1, col2 = st.columns(2)
    with col1:
        theta0 = st.number_input("θ₀ (rad)", value=1.0, step=0.5, format="%.2f")
        theta0_deg = theta0 * 180 / np.pi
        st.caption(f"≈ {theta0_deg:.1f}°")
    with col2:
        phi0 = st.number_input("φ₀ (rad)", value=1.0, step=0.5, format="%.2f")
        phi0_deg = phi0 * 180 / np.pi
        st.caption(f"≈ {phi0_deg:.1f}°")
    
    st.markdown("---")
    st.markdown("## 🧭 Dirección para Derivada")
    
    col1, col2 = st.columns(2)
    with col1:
        v_theta = st.number_input("v_θ", value=1.0, step=0.5, format="%.2f")
    with col2:
        v_phi = st.number_input("v_φ", value=0.5, step=0.5, format="%.2f")
    
    st.markdown("---")
    st.markdown("## 🎨 Visualización")
    
    resolucion = st.slider("Resolución de la malla:", 20, 100, 50)

# ============================================
# CÁLCULO DEL TENSOR MÉTRICO DEL TORO
# ============================================

@st.cache_data
def calcular_tensor_metrico(R, r):
    """Calcula el tensor métrico del toro en coordenadas (θ, φ)"""
    
    theta, phi = sp.symbols('theta phi', real=True)
    g_tt = sp.sympify(r**2)
    g_pp = sp.sympify((R + r * sp.cos(theta))**2)
    g_tp = sp.sympify(0)
    g = sp.Matrix([[g_tt, g_tp], [g_tp, g_pp]])
    g_inv = g.inv()
    det_g = sp.simplify(g.det())
    return {
        'g': g, 'g_inv': g_inv,
        'g_tt': g_tt, 'g_pp': g_pp,
        'det_g': det_g,
        'theta': theta, 'phi': phi
    }

# ============================================
# CÁLCULO DE LA FUNCIÓN Y DERIVADAS
# ============================================

@st.cache_data
def calcular_funcion(funcion_str, R, r, theta0, phi0, v_theta, v_phi):
    """Calcula todo: función, diferencial, gradiente, derivada direccional"""
    
    theta, phi = sp.symbols('theta phi', real=True)
    expr_str = funcion_str.replace('^', '**')
    try:
        namespace = {
            'theta': theta, 'phi': phi,
            'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
            'exp': sp.exp, 'log': sp.log, 'sqrt': sp.sqrt,
            'pi': sp.pi
        }
        f = sp.sympify(expr_str, locals=namespace)
        df_dtheta = sp.diff(f, theta)
        df_dphi = sp.diff(f, phi)
        metric = calcular_tensor_metrico(R, r)
        g = metric['g']
        g_inv = metric['g_inv']
        g_tt = metric['g_tt']
        g_pp = metric['g_pp']
        grad_theta = g_inv[0,0] * df_dtheta
        grad_phi = g_inv[1,1] * df_dphi
        subs = {theta: theta0, phi: phi0}
        f_num = float(f.subs(subs).evalf())
        df_dtheta_num = float(df_dtheta.subs(subs).evalf())
        df_dphi_num = float(df_dphi.subs(subs).evalf())
        g_tt_num = float(g_tt.subs({theta: theta0}).evalf())
        g_pp_num = float(g_pp.subs({theta: theta0}).evalf())
        grad_theta_num = float(grad_theta.subs(subs).evalf())
        grad_phi_num = float(grad_phi.subs(subs).evalf())
        norm = np.sqrt(v_theta**2 + v_phi**2)
        if norm > 0:
            v_theta_n = v_theta / norm
            v_phi_n = v_phi / norm
        else:
            v_theta_n, v_phi_n = 1.0, 0.0
        deriv_dir_analitica = grad_theta_num * v_theta_n + grad_phi_num * v_phi_n
        h = 0.0001
        f_original = f_num
        f_desp = float(f.subs({theta: theta0 + h*v_theta_n, 
                                phi: phi0 + h*v_phi_n}).evalf())
        deriv_dir_numerica = (f_desp - f_original) / h
        x_sym = (R + r * sp.cos(theta)) * sp.cos(phi)
        y_sym = (R + r * sp.cos(theta)) * sp.sin(phi)
        z_sym = r * sp.sin(theta)
        x_num = float(x_sym.subs({theta: theta0, phi: phi0}).evalf())
        y_num = float(y_sym.subs({theta: theta0, phi: phi0}).evalf())
        z_num = float(z_sym.subs({theta: theta0, phi: phi0}).evalf())
        return {
            'f': f,
            'df_dtheta': df_dtheta,
            'df_dphi': df_dphi,
            'grad_theta': grad_theta,
            'grad_phi': grad_phi,
            'f_num': f_num,
            'df_dtheta_num': df_dtheta_num,
            'df_dphi_num': df_dphi_num,
            'g_tt_num': g_tt_num,
            'g_pp_num': g_pp_num,
            'grad_theta_num': grad_theta_num,
            'grad_phi_num': grad_phi_num,
            'deriv_dir_analitica': deriv_dir_analitica,
            'deriv_dir_numerica': deriv_dir_numerica,
            'v_theta_n': v_theta_n,
            'v_phi_n': v_phi_n,
            'x_num': x_num,
            'y_num': y_num,
            'z_num': z_num,
            'theta0': theta0,
            'phi0': phi0
        }
    except Exception as e:
        st.error(f"Error en la función: {str(e)}")
        return None

# ============================================
# FUNCIÓN DE VISUALIZACIÓN 3D DEL TORO
# ============================================

def graficar_toro_con_funcion(R, r, funcion_str, resolucion, theta0, phi0):
    """Genera visualización 3D del toro coloreado por la función f(θ, φ)"""
    
    theta_vals = np.linspace(0, 2*np.pi, resolucion)
    phi_vals = np.linspace(0, 2*np.pi, resolucion)
    THETA, PHI = np.meshgrid(theta_vals, phi_vals)
    X = (R + r * np.cos(THETA)) * np.cos(PHI)
    Y = (R + r * np.cos(THETA)) * np.sin(PHI)
    Z = r * np.sin(THETA)
    expr_str = funcion_str.replace('^', '**')
    try:
        F = eval(expr_str, {
            'theta': THETA, 'phi': PHI, 'np': np,
            'sin': np.sin, 'cos': np.cos, 'exp': np.exp,
            'sqrt': np.sqrt, 'log': np.log, 'tan': np.tan
        })
    except Exception:
        F = np.cos(THETA) * np.cos(PHI)
    fig = go.Figure(data=[
        go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=F,
            colorscale=DEMETER_COLORSCALE,
            opacity=0.9,
            colorbar=dict(title="f(θ, φ)", x=1.05)
        )
    ])
    x0 = (R + r * np.cos(theta0)) * np.cos(phi0)
    y0 = (R + r * np.cos(theta0)) * np.sin(phi0)
    z0 = r * np.sin(theta0)
    fig.add_trace(go.Scatter3d(
        x=[x0], y=[y0], z=[z0],
        mode='markers',
        marker=dict(size=8, color='red', symbol='circle'),
        name=f'P({theta0:.2f}, {phi0:.2f})'
    ))
    fig.update_layout(
        title=f'Toro (R={R}, r={r}) coloreado por f(θ, φ) = {funcion_str[:40]}',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=700,
        height=600,
        margin=dict(l=0, r=0, b=0, t=50)
    )
    return fig


def graficar_toro_mpl(R, r, funcion_str, resolucion, theta0, phi0):
    """Versión matplotlib del toro (estática)"""
    
    theta_vals = np.linspace(0, 2*np.pi, resolucion)
    phi_vals = np.linspace(0, 2*np.pi, resolucion)
    THETA, PHI = np.meshgrid(theta_vals, phi_vals)
    X = (R + r * np.cos(THETA)) * np.cos(PHI)
    Y = (R + r * np.cos(THETA)) * np.sin(PHI)
    Z = r * np.sin(THETA)
    expr_str = funcion_str.replace('^', '**')
    try:
        F = eval(expr_str, {
            'theta': THETA, 'phi': PHI, 'np': np,
            'sin': np.sin, 'cos': np.cos, 'exp': np.exp,
            'sqrt': np.sqrt, 'log': np.log, 'tan': np.tan
        })
    except Exception:
        F = np.cos(THETA) * np.cos(PHI)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    f_norm = (F - np.nanmin(F)) / (np.nanmax(F) - np.nanmin(F) + 1e-9)
    surf = ax.plot_surface(X, Y, Z, facecolors=DEMETER_CMAP(f_norm), 
                           alpha=0.8, rstride=1, cstride=1)
    x0 = (R + r * np.cos(theta0)) * np.cos(phi0)
    y0 = (R + r * np.cos(theta0)) * np.sin(phi0)
    z0 = r * np.sin(theta0)
    ax.scatter([x0], [y0], [z0], color='red', s=100, label=f'P({theta0:.2f}, {phi0:.2f})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Toro (R={R}, r={r}) - f(θ, φ) = {funcion_str[:40]}')
    ax.legend()
    mappable = plt.cm.ScalarMappable(cmap=DEMETER_CMAP)
    mappable.set_array(F)
    fig.colorbar(mappable, ax=ax, shrink=0.5, label='f(θ, φ)')
    return fig

# ============================================
# COLUMNAS PRINCIPALES
# ============================================

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("## 📜 Resultados Matemáticos")
    metric = calcular_tensor_metrico(R, r)
    with st.expander("📐 Tensor Métrico del Toro", expanded=True):
        st.markdown(f"""
        <div class="card">
        <b>Coordenadas toroidales:</b> (θ, φ)<br>
        <b>Parametrización:</b><br>
        <code>x = (R + r·cos θ) · cos φ</code><br>
        <code>y = (R + r·cos θ) · sin φ</code><br>
        <code>z = r·sin θ</code><br><br>
        <b>Tensor métrico covariante g_ij:</b><br>
        <code>g_θθ = r² = {r}² = {r**2}</code><br>
        <code>g_φφ = (R + r·cos θ)²</code><br>
        <code>g_θφ = g_φθ = 0</code><br><br>
        <b>Matricialmente:</b><br>
        <code>⎡ {r**2}       0          ⎤</code><br>
        <code>⎢                          ⎥</code><br>
        <code>⎣ 0     (R + r·cos θ)²     ⎦</code><br><br>
        <b>Elemento de línea ds² = r² dθ² + (R + r·cos θ)² dφ²</b>
        </div>
        """, unsafe_allow_html=True)
    resultados = calcular_funcion(funcion_str, R, r, theta0, phi0, v_theta, v_phi)
    if resultados:
        with st.expander("📝 Diferencial Total df", expanded=True):
            st.markdown(f"""
            <div class="card">
            <b>df = (∂f/∂θ) dθ + (∂f/∂φ) dφ</b><br><br>
            ∂f/∂θ = <code>{resultados['df_dtheta']}</code><br>
            ∂f/∂φ = <code>{resultados['df_dphi']}</code><br><br>
            <b>En el punto P({theta0:.4f}, {phi0:.4f}):</b><br>
            ∂f/∂θ = <code>{resultados['df_dtheta_num']:.10f}</code><br>
            ∂f/∂φ = <code>{resultados['df_dphi_num']:.10f}</code>
            </div>
            """, unsafe_allow_html=True)
        with st.expander("∇ Gradiente en Coordenadas Toroidales", expanded=True):
            st.markdown(f"""
            <div class="card">
            <b>∇f = g^θθ (∂f/∂θ) e_θ + g^φφ (∂f/∂φ) e_φ</b><br><br>
            ∇f^θ = <code>{resultados['grad_theta']}</code><br>
            ∇f^φ = <code>{resultados['grad_phi']}</code><br><br>
            <b>En el punto P({theta0:.4f}, {phi0:.4f}):</b><br>
            ∇f^θ = <code>{resultados['grad_theta_num']:.10f}</code><br>
            ∇f^φ = <code>{resultados['grad_phi_num']:.10f}</code><br><br>
            <b>Tensor métrico en P:</b><br>
            g_θθ = <code>{resultados['g_tt_num']:.10f}</code><br>
            g_φφ = <code>{resultados['g_pp_num']:.10f}</code>
            </div>
            """, unsafe_allow_html=True)
        with st.expander("🎯 Derivada Direccional", expanded=True):
            st.markdown(f"""
            <div class="card">
            <b>Dirección:</b> v = ({v_theta}, {v_phi})<br>
            <b>Vector unitario:</b> v̂ = ({resultados['v_theta_n']:.6f}, {resultados['v_phi_n']:.6f})<br><br>
            <b>D_v̂ f = ∇f · v̂ = ∇f^θ v̂_θ + ∇f^φ v̂_φ</b><br><br>
            <b>Resultado analítico:</b> <code style="color:#FFD700; font-size:1.1rem;">{resultados['deriv_dir_analitica']:.10f}</code><br>
            <b>Verificación numérica:</b> <code>{resultados['deriv_dir_numerica']:.10f}</code><br>
            <b>Error absoluto:</b> <code>{abs(resultados['deriv_dir_analitica'] - resultados['deriv_dir_numerica']):.2e}</code>
            </div>
            """, unsafe_allow_html=True)
        with st.expander("🔄 Transformación Toro ↔ R3", expanded=True):
            st.markdown(f"""
            <div class="card">
            <b>Punto en coordenadas toroidales:</b><br>
            θ = {theta0:.6f} rad ≈ {theta0*180/np.pi:.2f}°<br>
            φ = {phi0:.6f} rad ≈ {phi0*180/np.pi:.2f}°<br><br>
            <b>↓ Transformación a R3 ↓</b><br>
            x = (R + r·cos θ)·cos φ = <code>{resultados['x_num']:.6f}</code><br>
            y = (R + r·cos θ)·sin φ = <code>{resultados['y_num']:.6f}</code><br>
            z = r·sin θ = <code>{resultados['z_num']:.6f}</code><br><br>
            <b>Punto en R3: P({resultados['x_num']:.6f}, {resultados['y_num']:.6f}, {resultados['z_num']:.6f})</b><br><br>
            <b>f(P) =</b> <code>{resultados['f_num']:.10f}</code>
            </div>
            """, unsafe_allow_html=True)
        with st.expander("📊 Evaluación de f(θ, φ)", expanded=True):
            st.markdown(f"""
            <div class="card">
            <b>f(θ, φ) =</b> <code>{resultados['f']}</code><br><br>
            <b>En el punto P({theta0:.4f}, {phi0:.4f}):</b><br>
            f(P) = <code>{resultados['f_num']:.10f}</code>
            </div>
            """, unsafe_allow_html=True)

with col2:
    st.markdown("## 📈 Visualización 3D del Toro")
    tab1, tab2 = st.tabs(["🍩 Toro Interactivo", "📊 Toro Estático"])
    with tab1:
        st.markdown("### Toro coloreado por f(θ, φ)")
        fig_interactivo = graficar_toro_con_funcion(R, r, funcion_str, resolucion, theta0, phi0)
        st.plotly_chart(fig_interactivo, width='stretch')
    with tab2:
        st.markdown("### Vista estática (Matplotlib)")
        fig_mpl = graficar_toro_mpl(R, r, funcion_str, resolucion, theta0, phi0)
        st.pyplot(fig_mpl)
        plt.close(fig_mpl)

st.markdown("---")
st.markdown("## 💡 Conclusión Física y Geométrica")

with st.container():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #8B0000, #4a0a2a); border-radius: 15px; padding: 1.5rem;">
    <p style="font-size: 1.1rem; line-height: 1.6;">
    <b style="color: #FFD700;">🍩 EL TENSOR MÉTRICO EN EL TORO</b>
    </p>
    <ul style="font-size: 1rem; line-height: 1.8;">
        <li><b style="color: #FFD700;">📐 Geometría no trivial:</b> El toro tiene curvatura variable. El tensor métrico <code>g_ij = diag(r², (R+r·cos θ)²)</code> captura esta geometría.</li>
        <li><b style="color: #FFD700;">🔄 Diferencial vs Gradiente:</b> El diferencial df = (∂f/∂θ)dθ + (∂f/∂φ)dφ es un <b>covector</b> (índice abajo). El gradiente ∇f = g^ij ∂_j f es un <b>vector</b> (índice arriba).</li>
        <li><b style="color: #FFD700;">⚡ La métrica "corrige" las escalas:</b> En el toro, ∇f^θ = (1/r²)·∂f/∂θ y ∇f^φ = 1/(R+r·cos θ)²·∂f/∂φ. Los factores dependen de la posición.</li>
        <li><b style="color: #FFD700;">🌍 Significado físico:</b> En una superficie curva, para medir longitudes y ángulos correctamente, la métrica es esencial. Sin ella, el concepto de "gradiente" como vector no está bien definido.</li>
    </ul>
    <p style="font-size: 1rem; font-style: italic; text-align: center; margin-top: 1rem;">
    "El toro nos enseña que la geometría no es plana: la métrica nos permite navegar por sus curvas."<br>
    — <b>Geometría Diferencial</b>
    </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align: center; margin-top: 2rem; padding: 1rem; color: #7a6a8a;">
<hr>
🍩 Cálculo Variacional y Tensorial | Superficies Curvas - El Toro 🍩
</div>
""", unsafe_allow_html=True)
