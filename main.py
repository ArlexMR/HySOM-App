"""
HySOM Streamlit App - Hysteresis Loop Classification Demo
This app demonstrates the usage of a self-organizing map (SOM) trained on 
discharge-concentration hysteresis loops for automatic classification of 
hydrologic events.
"""

import streamlit as st
import pathlib
from data_models import UserData, Loop
import pandas as pd
import time
from utils import (
    classify_loops,
    plot_frequency_map,
    plot_hysteresis_loops,
    plot_loops,
    create_time_series_plot,
    load_qt_data_from_file,
    load_events_data_from_file,
    calculate_dataset_metrics,
    extract_loops,
    build_classification_df,
    style_df,
    get_prototype
)
DATETIME_STR_FORMAT = "YYYY-MM-DD HH:mm:ss"

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
    
    # st.markdown("### 🔄 Reset Application")
    # if st.button("Clear All Data", width="stretch"):

    #     st.rerun()
    
    # st.divider()
    

    st.markdown("**Version**: 1.0.0 (Demo)")
    st.markdown("**Status**: Mock Functions Active")

# Initialize session state

if 'user_data' not in st.session_state:
    st.session_state.user_data = UserData(QC = None, events= None)
if 'classified_events' not in st.session_state:
    st.session_state.classified_loops = []


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
    som_image_path = pathlib.Path("assets").joinpath("TQSOM_large.jpg")
    if som_image_path.exists():
        st.image(str(som_image_path), 
                 width=890)
    else:
        st.error("Failed to load the General T-Q SOM image", icon=":material/broken_image:")

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
    if st.button("📂 Load Example Data", width="stretch"):
        try:
            # Load example data
            qc_path = pathlib.Path("assets").joinpath("QTdata.csv")
            events_path = pathlib.Path("assets").joinpath("events.csv")
            if qc_path.exists() and events_path.exists():
                qc = load_qt_data_from_file(qc_path)
                user_events = load_events_data_from_file(events_path)
                st.session_state.user_data.QC = qc
                st.session_state.user_data.events = user_events
                scs = st.success("✅ Example data loaded successfully!")
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
    with st.spinner("Extracting loops...", show_time=True):
        loops = extract_loops(qc, user_events)
    scs1 = st.success(f"✅ Succesfully extracted {len(loops)} loops.")

    with st.spinner("Classifying loops...", show_time=True):
        classified_loops = classify_loops(loops)
    scs2  = st.success(f"✅ Succesfully classified {len(classified_loops)} loops.")
    st.session_state.classified_loops = classified_loops
    scs1.empty()
    time.sleep(1.0)
    scs2.empty()


classified_loops = st.session_state.classified_loops

# ==================== METRICS ====================
st.markdown("### 📊 Dataset Summary")

# Calculate and display metrics
metrics = calculate_dataset_metrics(qc, classified_loops)

cols = st.columns(len(metrics))
for col, (label, value) in zip(cols, metrics.items()):
    with col:
        st.metric(label=label, value=value,)

st.divider()

# ==================== TIME SERIES VISUALIZATION ====================
st.markdown("### 📈 Time Series Visualization")

# Create interactive plotly chart
# Create interactive plotly chart
with st.spinner("Creating time series chart...", show_time=True):
    fig = create_time_series_plot(qc, user_events)

st.plotly_chart(fig, width = "stretch")

st.info("💡 **Tip**: Green shaded areas represent hydrologic events. Zoom in to explore specific events in detail!")

st.divider()

# ==================== EVENTS TABLE & LOOPS VIEWER ====================
st.markdown("### 📋 Event Classifications & Loop Viewer")

st.markdown("""
This table shows each hydrologic event with its BMU coordinates and distance metric.
Select events using checkboxes to plot their hysteresis loops.
""")

# Wrap entire section in fragment to prevent full app reruns
@st.fragment
def display_events_table_and_loop_viewer():
    # Two-column layout: Table on left, Comparison chart on right
    col_table, col_comparison = st.columns([1.2, 1])
    classified_loops = st.session_state.classified_loops
    with col_table:
        st.markdown("#### Events Table")
        
        classification_df = build_classification_df(classified_loops)
        display_df = style_df(classification_df, 
                              cmap = "RdYlGn_r",
                              vmin=-0.5,
                              vmax=3.5,
                              alpha=0.1,
                              subset = ["distance"]
                              )
        df_selection = st.dataframe(
            display_df,
            width="stretch",
            height = 355,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "ID": st.column_config.NumberColumn("Event #", format="%d", pinned=True),
                "start": st.column_config.DatetimeColumn("Start Time", format=DATETIME_STR_FORMAT),
                "end": st.column_config.DatetimeColumn("End Time", format=DATETIME_STR_FORMAT),
                "BMU" : st.column_config.TextColumn("BMU [row, col]", help = "Coordinates of the Best Matching Unit, i.e., the prototype in the General T-Q SOM that is most similar to the given loop."),
                "distance": st.column_config.NumberColumn("Distance", format="%.4f", help="DTW distance from BMU prototype (lower is better)")
            }
        )
        
        # Download button for results
        csv = classification_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="hysteresis_classification_results.csv",
            mime="text/csv",
            width="stretch"
        )
    
    with col_comparison:
        st.markdown("#### Loop Viewer")
        add_prototype = st.button("➕Add Prototype")
        selected_row = df_selection.get("selection", {}).get("rows", [])

        # Collect loops and labels
        loops_to_plot = []
        labels_to_plot = []

        # Add selected event loops
        classified_loops = st.session_state.classified_loops
        loops_to_plot = []
        labels_to_plot = []
        if selected_row:
            row_id = selected_row[0]
            loop_id = classification_df.iloc[row_id]["ID"] 
            loop = next((item for item in classified_loops if item.ID == loop_id))
            loop_label = f"Event {loop_id}"
            loops_to_plot.append(loop.coordinates)
            labels_to_plot.append(loop_label)
            if add_prototype:
                bmu_row, bmu_col = classification_df.iloc[row_id]["BMU"]
                proto_loop = get_prototype((bmu_row, bmu_col))
                proto_label = f"Prototype ({bmu_row}, {bmu_col})"
                loops_to_plot.append(proto_loop)
                labels_to_plot.append(proto_label)

        # Add prototype loops

    
        # Plot comparison
        if len(loops_to_plot) > 0:
            comp_fig = plot_loops(
                loops_to_plot,
                labels_to_plot,
                title=f"Selected Loop"
            )
            st.plotly_chart(comp_fig, width=300)
            
            # Clear buttons
            col_clear1, col_clear2 = st.columns(2)
            with col_clear1:
                if st.button("Clear Prototype", width="stretch"):
                    st.session_state.selected_prototypes = set()
                    st.rerun(scope="fragment")  # Only rerun the fragment, not the entire app
        else:
            st.info("Select events from the table or add a prototype to compare loops")

# Call the fragment
classified_loops = st.session_state.classified_loops
if len(classified_loops) > 0:
    display_events_table_and_loop_viewer()
else:
    st.warning("No valid loops found")

st.divider()

# ==================== FREQUENCY MAP & LOOP VIEWER ====================
st.markdown("### 🗺️ Frequency Distribution & Loop Explorer")

st.markdown("""
The heatmap shows how many hysteresis loops map to each SOM unit. 
Select a BMU coordinate to visualize the loops for that prototype.
""")

col_freq_map, col_bmu_selector_and_loops = st.columns([5, 7])
classified_loops = st.session_state.classified_loops
with col_freq_map:
    st.markdown("#### Frequency Heatmap")
    with st.spinner("Generating Frequency Distribution...", show_time=True):
        freq_fig = plot_frequency_map(classified_loops)
        st.pyplot(freq_fig, width="stretch")

with col_bmu_selector_and_loops:
    st.markdown("#### Hysteresis Loop Viewer")
    
    # Use fragment to prevent full app rerun when selecting coordinates
    @st.fragment
    def loop_viewer_fragment():
        # BMU selector
        col_bmu_selector, col_loops_chart = st.columns([2,5])
        with col_bmu_selector:
            with st.form("BMU selection"):
                # subcol1, subcol2, subcol3 = st.columns(3)

                selected_row = st.selectbox(
                    "BMU Row",
                    options=list(range(8)),
                    index=0,
                    help="Row coordinate (0-7)",
                    key="bmu_row_selector"
                )
                
                # with subcol2:
                selected_col = st.selectbox(
                    "BMU Column",
                    options=list(range(8)),
                    index=0,
                    help="Column coordinate (0-7)",
                    key="bmu_col_selector"
                )
                submitted_bmu = st.form_submit_button("Plot Loops")    

            classified_loops = st.session_state.classified_loops
            matching_loops = [loop for loop in classified_loops if loop.BMU == (selected_row, selected_col)]

            st.metric(
                label=f"Events at ({selected_row}, {selected_col})",
                value=len(matching_loops)
        )
        # if not submitted_bmu:
        #     st.info("Select a BMU to see its associated loops")
        #     st.stop()
        if not matching_loops:
            with col_loops_chart:
                st.info("No loops for the selected BMU")
            st.stop()
            
        
        # if matching_loops:
        selected_loops, ids = zip(*[(loop.coordinates, loop.ID) for loop in matching_loops])
        # Show count of events at this BMU
        with col_loops_chart:
            if len(selected_loops) > 0:
                # Plot the loops
                loop_fig = plot_hysteresis_loops(list(selected_loops), list(ids), selected_row, selected_col)

                st.plotly_chart(loop_fig, width="stretch")
                
                # Show event IDs
                st.caption(f"**Events:** {', '.join([f'E{eid}' for eid in ids])}")
            else:
                st.info(f"No events at BMU ({selected_row}, {selected_col})")
    
    # Call the fragment
    loop_viewer_fragment()


