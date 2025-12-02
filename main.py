"""
HySOM Streamlit App - Hysteresis Loop Classification Demo
This app demonstrates the usage of a self-organizing map (SOM) trained on 
discharge-concentration hysteresis loops for automatic classification of 
hydrologic events.
"""

import streamlit as st
import pathlib
from data_models import UserData, Loop
from utils import (
    classify_loops,
    create_frequency_map,
    plot_frequency_map,
    get_loops_for_bmu,
    plot_hysteresis_loops,
    get_average_loop_for_bmu,
    plot_loop_comparison,
    create_time_series_plot,
    load_qt_data_from_file,
    load_events_data_from_file,
    calculate_dataset_metrics,
    extract_loops
)

# Page configuration
st.set_page_config(
    page_title="HySOM - Hysteresis Loop Classifier",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.html("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1.5rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    </style>
""")

# Initialize session state

if 'user_data' not in st.session_state:
    st.session_state.user_data = UserData(QC = None, events= None)
if 'classified_events' not in st.session_state:
    st.session_state.classified_loops = []
st.write(f"number of elements in session_state.classified_events: {len(st.session_state.classified_loops)}")


# ==================== HEADER ====================
st.html('<div class="main-header">🌊 HySOM: Hysteresis Loop Classifier</div>')
st.html('<div class="sub-header">Automatic classification of suspended sediment hysteresis loops using Self-Organizing Maps</div>')

# ==================== INTRODUCTION ====================
with st.expander("ℹ️ About this Application", expanded=False):
    st.markdown("""
    ### What is this app?
    
    This application demonstrates the use of a **Self-Organizing Map (SOM)** trained on discharge-concentration 
    hysteresis loops to automatically classify hydrologic events. The trained SOM allows researchers to:
    
    - 📊 **Classify hysteresis patterns** from watershed events
    - 🔍 **Identify similar hydrologic behaviors** across different events
    - 📈 **Visualize frequency distributions** of loop types in your watershed
    
    ### How does it work?
    
    1. **Upload your data**: Provide discharge-concentration time series and event definitions
    2. **Automatic processing**: The app extracts hysteresis loops for each event
    3. **Classification**: Each loop is mapped to its Best Matching Unit (BMU) on the trained SOM
    4. **Visualization**: View the frequency distribution and event classifications
    
    ### The General T-Q SOM
    
    The SOM used in this application was trained on a diverse set of hysteresis loops from multiple watersheds,
    making it a "general" classifier that can be applied to new data.
    """)

# ==================== SOM VISUALIZATION ====================
st.markdown("### 🗺️ The Trained Self-Organizing Map")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    som_image_path = pathlib.Path("assets").joinpath("TQSOM.png")
    if som_image_path.exists():
        st.image(str(som_image_path), caption="General T-Q SOM: Each cell represents a distinct hysteresis pattern", use_container_width=True)
    else:
        st.info("SOM visualization image not found. Please ensure 'assets/TQSOM.png' exists.")

st.divider()

# ==================== DATA UPLOAD ====================
st.markdown("### 📁 Upload Your Data")

col1, col2 = st.columns(2)

# Load QC data
with col1:
    st.markdown("#### Discharge-Concentration Time Series")
    uploaded_qc = st.file_uploader(
        "Upload CSV file with columns: `datetime`, `Qcms`, `turb`",
        type=["csv"],
        key="qt_uploader",
        help="CSV file containing timestamp, discharge (Qcms), and concentration (turb) data"
    )
    
    if uploaded_qc is not None:
        try:
            qc = load_qt_data_from_file(uploaded_qc)
            st.session_state.user_data.QC = qc
            st.success(f"✅ Loaded {len(qc)} time series records")
            
            # Show data preview
            with st.expander("Preview data"):
                st.dataframe(qc.head(max(10, len(qc))), width="stretch")
        except Exception as e:
            st.error(f"Error loading Q-T data: {str(e)}")

# Load Events data
with col2:
    st.markdown("#### Hydrologic Events")
    uploaded_events = st.file_uploader(
        "Upload CSV file with columns: `start`, `end`",
        type=["csv"],
        key="events_uploader",
        help="CSV file containing start and end timestamps for each hydrologic event"
    )
    
    if uploaded_events is not None:
        try:
            user_events = load_events_data_from_file(uploaded_events)
            st.session_state.user_data.events = user_events
            st.success(f"✅ Loaded {len(user_events)} events")
            
            # Show data preview
            with st.expander("Preview events"):
                st.dataframe(user_events.head(max(10, len(user_events))), width="stretch")
        except Exception as e:
            st.error(f"Error loading events data: {str(e)}")

# Load example data button
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("📂 Load Example Data", use_container_width=True):
        try:
            # Load example data
            qc_path = pathlib.Path("assets").joinpath("QTdata.csv")
            events_path = pathlib.Path("assets").joinpath("events.csv")
            if qc_path.exists() and events_path.exists():
                qc = load_qt_data_from_file(qc_path)
                user_events = load_events_data_from_file(events_path)
                st.session_state.user_data.QC = qc
                st.session_state.user_data.events = user_events
                st.success("✅ Example data loaded successfully!")
            else:
                st.error("Example data files not found in assets folder")
        except Exception as e:
            st.error(f"Error loading example data: {str(e)}")

st.divider()

# ==================== ANALYSIS & VISUALIZATION ====================

if (st.session_state.user_data.QC is None) or (st.session_state.user_data.events is None):
    st.info("👆 Please upload your data files or load the example data to begin the analysis.")
    st.stop()

qc = st.session_state.user_data.QC
user_events = st.session_state.user_data.events

# classify events if not already done
if not st.session_state.classified_loops:
    # Show progress bar for processing   
    progress_bar = st.progress(0)

    # Step 1: Extract loops
    progress_bar.progress(0, text="📊 Extracting hysteresis loops from events...")
    loops = extract_loops(qc, user_events)
    
    # Step 2: Classify loops
    progress_bar.progress(50, text="🧮 Mapping loops onto the General T-Q SOM...")
    classified_loops = classify_loops(loops)
    
    # Step 3: Complete
    progress_bar.progress(100, text="✅ Processing complete!")
    st.session_state.classified_loops = classified_loops
    
    # Clear progress bar after a brief moment
    import time
    time.sleep(0.5)
    progress_bar.empty()

classified_loops = st.session_state.classified_loops

# ==================== METRICS ====================
st.markdown("### 📊 Dataset Summary")

# Calculate and display metrics
metrics = calculate_dataset_metrics(qc, classified_loops)

cols = st.columns(len(metrics))
for col, (label, value) in zip(cols, metrics.items()):
    with col:
        st.metric(label=label, value=value)

st.divider()

# ==================== TIME SERIES VISUALIZATION ====================
st.markdown("### 📈 Time Series Visualization")

# Create interactive plotly chart
# Create interactive plotly chart
fig = create_time_series_plot(qc, user_events)

st.plotly_chart(fig, width = "stretch")

st.info("💡 **Tip**: Green shaded areas represent hydrologic events. Zoom in to explore specific events in detail!")

st.divider()

# ==================== EVENTS TABLE & LOOP COMPARISON ====================
st.markdown("### 📋 Event Classifications & Loop Comparison")

st.markdown("""
This table shows each hydrologic event with its BMU coordinates and distance metric.
Select events using checkboxes to plot their hysteresis loops.
""")

# Wrap entire section in fragment to prevent full app reruns
@st.fragment
def events_and_comparison_fragment():
    # Two-column layout: Table on left, Comparison chart on right
    col_table, col_comparison = st.columns([1.2, 1])
    
    with col_table:
        st.markdown("#### Events Table")
        
        # Format the table for display
        display_df = bmu_results.copy()
        # display_df['start'] = display_df['start'].dt.strftime('%Y-%m-%d %H:%M')
        # display_df['end'] = display_df['end'].dt.strftime('%Y-%m-%d %H:%M')
        
        # Display with selection (on_select="rerun" only reruns the fragment now)
        event_selection = st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row",
            column_config={
                "Event_ID": st.column_config.NumberColumn("Event #", format="%d"),
                # "start": st.column_config.TextColumn("Start Time"),
                # "end": st.column_config.TextColumn("End Time"),
                "BMU_Row": st.column_config.NumberColumn("BMU Row", format="%d"),
                "BMU_Col": st.column_config.NumberColumn("BMU Col", format="%d"),
                "Distance": st.column_config.NumberColumn("Distance", format="%.4f", help="Distance from BMU prototype (lower is better)")
            }
        )
        
        # Download button for results
        csv = bmu_results.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="hysteresis_classification_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_comparison:
        st.markdown("#### Loop Comparison")
        
        # Get selected rows
        selected_rows = event_selection.selection.rows if event_selection.selection else [] # type: ignore
        
        # Controls for adding prototype
        with st.expander("➕ Add SOM Prototype Loop"):
            subcol1, subcol2, subcol3 = st.columns([1, 1, 1])
            with subcol1:
                proto_row = st.number_input("Row", min_value=0, max_value=7, value=0, key="proto_row")
            with subcol2:
                proto_col = st.number_input("Col", min_value=0, max_value=7, value=0, key="proto_col")
            with subcol3:
                add_proto = st.button("Add", use_container_width=True, key="add_proto")
        
        # Initialize session state for prototype selection
        if 'selected_prototypes' not in st.session_state:
            st.session_state.selected_prototypes = []
        
        if add_proto:
            proto_key = (proto_row, proto_col)
            if proto_key not in st.session_state.selected_prototypes:
                st.session_state.selected_prototypes.append(proto_key)
        
        # Collect loops and labels
        loops_to_plot = []
        labels_to_plot = []
        
        # Add selected event loops
        if len(selected_rows) > 0:
            for loop_id in selected_rows:
                loop = loops_w_ids["loop_id"]
                # event_row = bmu_results.loc[row_idx]
                # event_id = event_row['Event_ID']
                # start = event_row['start']
                # end = event_row['end']
                
                # # Extract loop data
                # mask = (qt_data.index >= start) & (qt_data.index <= end)
                # event_data = qt_data[mask]
                
                # if len(event_data) > 0:
                #     q_values = event_data['Qcms'].values
                #     c_values = event_data['turb'].values
                #     loop = np.column_stack([q_values, c_values])
                loops_to_plot.append(loop)
                labels_to_plot.append(f"Event {loop_id}")
        
        # Add prototype loops
        for proto_key in st.session_state.selected_prototypes:
            avg_loop = get_average_loop_for_bmu(
                qc, user_events, bmu_results,
                proto_key[0], proto_key[1]
            )
            if len(avg_loop) > 0:
                loops_to_plot.append(avg_loop)
                labels_to_plot.append(f"Prototype ({proto_key[0]}, {proto_key[1]})")
        
        # Plot comparison
        if len(loops_to_plot) > 0:
            comp_fig = plot_loop_comparison(
                loops_to_plot,
                labels_to_plot,
                title=f"Comparing {len(loops_to_plot)} Loop(s)"
            )
            st.plotly_chart(comp_fig, use_container_width=True)
            
            # Clear buttons
            col_clear1, col_clear2 = st.columns(2)
            with col_clear1:
                if st.button("Clear Prototypes", use_container_width=True):
                    st.session_state.selected_prototypes = []
                    st.rerun(scope="fragment")  # Only rerun the fragment, not the entire app
        else:
            st.info("Select events from the table or add a prototype to compare loops")

# Call the fragment
events_and_comparison_fragment()

st.divider()

# ==================== FREQUENCY MAP & LOOP VIEWER ====================
st.markdown("### 🗺️ Frequency Distribution & Loop Explorer")

st.markdown("""
The heatmap shows how many hysteresis loops map to each SOM unit. 
Select a BMU coordinate to visualize the loops for that prototype.
""")

# Create frequency map (only once)
freq_map = create_frequency_map(bmu_results)

# Two-column layout: Frequency map on left, Loop viewer on right
col_freq, col_loops = st.columns([1, 1])

with col_freq:
    st.markdown("#### Frequency Heatmap")
    freq_fig = plot_frequency_map(freq_map, title="Event Distribution")
    st.pyplot(freq_fig, use_container_width=True)

with col_loops:
    st.markdown("#### Hysteresis Loop Viewer")
    
    # Use fragment to prevent full app rerun when selecting coordinates
    @st.fragment
    def loop_viewer_fragment():
        # BMU selector
        subcol1, subcol2 = st.columns(2)
        
        with subcol1:
            selected_row = st.selectbox(
                "BMU Row",
                options=list(range(8)),
                index=0,
                help="Row coordinate (0-7)",
                key="bmu_row_selector"
            )
        
        with subcol2:
            selected_col = st.selectbox(
                "BMU Column",
                options=list(range(8)),
                index=0,
                help="Column coordinate (0-7)",
                key="bmu_col_selector"
            )
        
        # Show count of events at this BMU
        count_at_bmu = len(bmu_results[
            (bmu_results['BMU_Row'] == selected_row) & 
            (bmu_results['BMU_Col'] == selected_col)
        ])
        
        st.metric(
            label=f"Events at ({selected_row}, {selected_col})",
            value=count_at_bmu
        )
        
        # Get and plot loops for selected BMU
        loops, event_ids = get_loops_for_bmu(
            qc, 
            user_events, 
            bmu_results, 
            selected_row, 
            selected_col
        )
        
        if len(loops) > 0:
            # Plot the loops
            loop_fig = plot_hysteresis_loops(loops, event_ids, selected_row, selected_col)
            st.plotly_chart(loop_fig, use_container_width=True)
            
            # Show event IDs
            st.caption(f"**Events:** {', '.join([f'E{eid}' for eid in event_ids])}")
        else:
            st.info(f"No events at BMU ({selected_row}, {selected_col})")
    
    # Call the fragment
    loop_viewer_fragment()




# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## ⚙️ Settings & Info")
    
    st.markdown("### SOM Configuration")
    st.markdown("""
    - **Grid Size**: 8 × 8
    - **Training Data**: Multi-watershed dataset
    - **Input Features**: Normalized Q-C loops
    """)
    
    st.divider()
    
    st.markdown("### 📚 Resources")
    st.markdown("""
    - [Documentation](#)
    - [GitHub Repository](#)
    - [Research Paper](#)
    """)
    
    st.divider()
    
    st.markdown("### 🔄 Reset Application")
    if st.button("Clear All Data", use_container_width=True):
        st.session_state.qt_data = None
        st.session_state.events_data = None
        st.session_state.bmu_results = None
        st.rerun()
    
    st.divider()
    
    st.markdown("---")
    st.markdown("**Version**: 1.0.0 (Demo)")
    st.markdown("**Status**: Mock Functions Active")
