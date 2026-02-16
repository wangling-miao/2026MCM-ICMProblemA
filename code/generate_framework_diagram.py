"""
Framework Diagram Generator (Final Polish)
==========================================
Generates a high-end academic framework diagram with strict alignment and modern aesthetics.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import os

def create_framework_diagram():
    # Setup figure
    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis('off')
    
    # Modern Palette (Flat UI)
    colors = {
        'col1_bg': '#eef2f3',  # Light Grey
        'col2_bg': '#e8f6f3',  # Mint
        'col3_bg': '#eaf2f8',  # Light Blue
        'col4_bg': '#f4ecf7',  # Light Purple
        
        'node_data': '#bdc3c7',
        'node_load': '#a2d9ce',
        'node_core': '#85c1e9', # Strong Blue
        'node_out':  '#d2b4de',
        
        'text_dark': '#2c3e50',
        'edge': '#34495e'
    }

    # 1. Draw Column Backgrounds
    # Col 1: Data
    ax.add_patch(patches.Rectangle((2, 2), 18, 46, color=colors['col1_bg'], ec='none', alpha=0.6, zorder=0))
    ax.text(11, 46, "I. Data Foundation", ha='center', fontsize=12, fontweight='bold', color='#7f8c8d')
    
    # Col 2: Load
    ax.add_patch(patches.Rectangle((22, 2), 20, 46, color=colors['col2_bg'], ec='none', alpha=0.6, zorder=0))
    ax.text(32, 46, "II. Load Modeling", ha='center', fontsize=12, fontweight='bold', color='#16a085')

    # Col 3: Physics Core
    ax.add_patch(patches.Rectangle((44, 2), 32, 46, color=colors['col3_bg'], ec='none', alpha=0.6, zorder=0))
    ax.text(60, 46, "III. Coupled Physics Core", ha='center', fontsize=12, fontweight='bold', color='#2980b9')

    # Col 4: Output
    ax.add_patch(patches.Rectangle((78, 2), 20, 46, color=colors['col4_bg'], ec='none', alpha=0.6, zorder=0))
    ax.text(88, 46, "IV. Estimation", ha='center', fontsize=12, fontweight='bold', color='#8e44ad')

    # Helper: Draw Node
    def node(x, y, w, h, title, sub, color):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.2', 
                                    facecolor=color, edgecolor='white', linewidth=2, zorder=10)
        ax.add_patch(rect)
        ax.text(x+w/2, y+h/2+1, title, ha='center', va='center', fontsize=10, fontweight='bold', color='#2c3e50', zorder=11)
        ax.text(x+w/2, y+h/2-1, sub, ha='center', va='center', fontsize=8, color='#444', zorder=11)
        return (x, y, w, h)

    def link(p1, p2, label=None, bend=False):
        # Draw arrow
        xy = p2
        xytext = p1
        
        con = "arc3,rad=0"
        if bend: con = "arc3,rad=-0.2"
        
        ax.annotate('', xy=xy, xytext=xytext, 
                    arrowprops=dict(arrowstyle='->,head_width=0.4', lw=2, color='#34495e', connectionstyle=con), zorder=5)
        
        if label:
            mx, my = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
            ax.text(mx, my+1, label, ha='center', fontsize=8, backgroundcolor='white', color='#c0392b', zorder=6)

    # --- Nodes ---
    
    # I. Data
    node(5, 35, 12, 6, "Experiment Data", "HPPC, OCV Curve", '#ffffff')
    node(5, 20, 12, 6, "Usage Trace", "App Logs, Current", '#ffffff')
    node(5, 5, 12, 6, "Aging Data", "Cycle Life Test", '#ffffff')

    # II. Load
    node(25, 30, 14, 8, "HMM Generator", "State Transition\n(Video->Game)", '#d1f2eb')
    node(25, 15, 14, 8, "Power Model", "$P_{total} = P_{cpu} + ...$", '#d1f2eb')
    
    # III. Core (The Complex Part)
    # Arrange in a triangle loop
    
    # 1. Electrical (Top)
    node(50, 32, 20, 8, "Electrical Model", "2nd-Order RC\n$V_{term} = f(SOC, I)$", '#a9cce3')
    
    # 2. Thermal (Right Bottom)
    node(60, 15, 12, 8, "Thermal Model", "$T_{cell}$ (Arrhenius)", '#a9cce3')
    
    # 3. Aging (Left Bottom)
    node(48, 15, 11, 8, "Aging Model", "SEI Growth ($Q_n$)", '#a9cce3')

    # IV. Output
    node(81, 32, 14, 8, "EKF Observer", "Voltage Correction\nSOC Estimation", '#e8daef')
    node(81, 15, 14, 8, "TTE Prediction", "Time-to-Empty\nDistribution", '#e8daef')

    # --- Links ---
    
    # Data -> Load
    link((17, 23), (25, 34), "Train") # Trace -> HMM
    link((17, 23), (25, 19)) # Trace -> Power
    
    # Data -> Core (Param ID)
    link((17, 38), (50, 38), "Param ID") # HPPC -> Elec
    
    # Load -> Power
    link((32, 30), (32, 23), "State") # HMM -> Power
    
    # Power -> Elec
    link((39, 19), (49, 34), "Current $I(t)$", bend=True)
    
    # Core Coupling Ring
    # Elec -> Thermal (Joules)
    ax.annotate('', xy=(66, 23), xytext=(66, 32), 
                arrowprops=dict(arrowstyle='->', lw=2, color='#e74c3c'), zorder=5)
    ax.text(67, 27, "$Q_{gen}$", color='#e74c3c', fontsize=9)
    
    # Thermal -> Aging (Temp)
    link((60, 19), (59, 19))
    
    # Thermal -> Elec (Param T)
    ax.annotate('', xy=(55, 32), xytext=(60, 23), 
                arrowprops=dict(arrowstyle='->', lw=2, color='#e67e22', connectionstyle="arc3,rad=-0.1"), zorder=5)
    ax.text(56, 27, "$R(T)$", color='#e67e22', fontsize=9)

    # Aging -> Elec (Capacity Fade)
    ax.annotate('', xy=(52, 32), xytext=(52, 23), 
                arrowprops=dict(arrowstyle='->', lw=2, color='#7f8c8d'), zorder=5)
    ax.text(53, 27, "$Q_n$", color='#7f8c8d', fontsize=9)

    # Core -> Out
    link((70, 36), (81, 36), "Voltage") # Elec -> EKF
    link((88, 32), (88, 23), "SOC")     # EKF -> TTE
    
    # Title
    fig.text(0.5, 0.95, "Figure 1: Multi-Physics Coupled System Architecture", 
             ha='center', fontsize=14, fontweight='bold', color='#2c3e50')

    # Save
    output_path = os.path.join("paper_results", "framework_diagram.png")
    os.makedirs("paper_results", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Polished diagram saved to {output_path}")

if __name__ == "__main__":
    create_framework_diagram()
