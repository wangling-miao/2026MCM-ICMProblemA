
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_flowchart(output_path):
    # set up figure - professional aesthetic (wide, clean)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Style constants - "Academic Paper" style
    # Black/White/Grays with a single accent color (Royal Blue)
    styles = {
        'container': dict(boxstyle='round,pad=0.05', facecolor='#F8F9FA', edgecolor='#AAAAAA', linestyle='--', linewidth=1.5),
        'block':     dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', linewidth=1.5),
        'start':     dict(boxstyle='round4,pad=0.5', facecolor='#E1E1E1', edgecolor='black', linewidth=1.5),
        'decision':  dict(boxstyle='darrow,pad=0.3', facecolor='white', edgecolor='black', linewidth=1.5), # Diamond shape via text logic approx or patches
        'accent':    dict(boxstyle='round,pad=0.4', facecolor='#E6F3FF', edgecolor='#0066CC', linewidth=2),
    }

    def draw_box(x, y, text, style_key, width=None, height=None, font_size=10):
        # We use ax.text for boxes as it handles text centering well.
        # For precise dimensions we might use patches, but text bbox is usually sufficient for flowcharts.
        # If width/height needed, we can pad text.
        
        st = styles[style_key]
        t = ax.text(x, y, text, ha='center', va='center', bbox=st, fontsize=font_size, color='black', zorder=10)
        return t

    # --- Layout Logic ---
    
    # 1. Input Block (Left)
    draw_box(0.08, 0.5, " Initialization \n$Q_{n0}, R_0(T), \dots$\n(N=200 Samples)", 'start', font_size=11)
    
    # 2. Results Block (Right)
    draw_box(0.92, 0.5, " Statistical Analysis \nPDF(TTE)\n90% CI", 'start', font_size=11)

    # 3. Main Solver Container (Middle)
    # Draw a large rectangle behind to represent the "Time-Stepping Loop"
    rect = patches.FancyBboxPatch((0.18, 0.1), 0.64, 0.8, boxstyle='round,pad=0.05', 
                                  linewidth=1.5, edgecolor='#666666', facecolor='#F5F5F5', linestyle='--')
    ax.add_patch(rect)
    ax.text(0.5, 0.86, "Hybrid Time-Stepping Solver (Time $t \to t+\Delta t$)", ha='center', fontsize=12, fontweight='bold', color='#444444')

    # Inside the Solver Container: Left-to-Right Flow
    
    # Step A: HMM
    x_start = 0.28
    y_row = 0.5
    spacing = 0.15
    
    # A: HMM
    draw_box(x_start, y_row, "1. User Behavior\n(HMM Sampling)", 'block', font_size=10)
    
    # B: Coupling
    draw_box(x_start + spacing, y_row, "2. Coupling\n$I_k = P_{load}/V_{k-1}$", 'block', font_size=10)
    
    # C: Dynamics (Accent this one as it's the core math)
    draw_box(x_start + 2*spacing, y_row, r"3. State Update" + "\n" + r"$\frac{dSOC}{dt}, \frac{d\delta}{dt}, V_{RC}$", 'accent', font_size=10)
    
    # D: Check
    # Using a text box for check
    draw_box(x_start + 3*spacing, y_row, "4. Cutoff Check\n$V < V_{cutoff}$?", 'block', font_size=10)

    # --- Arrows ---
    arrow_props = dict(arrowstyle='->', lw=1.5, color='black')
    
    # Init -> Solver Frame
    ax.annotate("", xy=(0.18, 0.5), xytext=(0.14, 0.5), arrowprops=arrow_props)
    
    # Solver Frame -> Results
    ax.annotate("", xy=(0.87, 0.5), xytext=(0.82, 0.5), arrowprops=arrow_props)
    
    # Inside Solver
    # HMM -> Coupling
    ax.annotate("", xy=(x_start + spacing*0.7, y_row), xytext=(x_start + spacing*0.35, y_row), arrowprops=arrow_props)
    # Coupling -> Dynamics
    ax.annotate("", xy=(x_start + spacing*1.7, y_row), xytext=(x_start + spacing*1.35, y_row), arrowprops=arrow_props)
    # Dynamics -> Check
    ax.annotate("", xy=(x_start + spacing*2.7, y_row), xytext=(x_start + spacing*2.35, y_row), arrowprops=arrow_props)

    # LOOP BACK (No Cutoff)
    # From Check "No" -> Back to Start of Solver (HMM)
    # Draw a line downwards and back
    ax.annotate("No (Next Step)", xy=(x_start, y_row - 0.1), xytext=(x_start + 3*spacing, y_row - 0.1),
                arrowprops=dict(arrowstyle="->", connectionstyle="bar,angle=180,fraction=-0.1", lw=1.5, ls='dashed'),
                ha='center', va='bottom', fontsize=9)
    # Connect Check bottom to this path
    ax.plot([x_start + 3*spacing, x_start + 3*spacing], [y_row - 0.05, y_row - 0.1], color='black', lw=1.5, ls='dashed')
    # Connect to HMM bottom
    ax.plot([x_start, x_start], [y_row - 0.1, y_row - 0.05], color='black', lw=1.5, ls='dashed')

    # EXIT (Yes Cutoff)
    # This is implicitly the arrow leaving the box to the right, but let's label it
    ax.text(0.84, 0.52, "Yes (Record)", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    output_file = r"C:\Users\chenp\Documents\比赛\2026美赛\code\paper_results\algorithm_flowchart.png"
    draw_flowchart(output_file)
    print(f"Flowchart generated at: {output_file}")
