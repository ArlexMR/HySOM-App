"""
Mock utility functions for the HySOM Streamlit App
These functions simulate the ML model operations for UI/UX demonstration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import List, Tuple, Dict
from datetime import datetime


def extract_hysteresis_loops(
    qt_data: pd.DataFrame, 
    events: pd.DataFrame
) -> List[np.ndarray]:
    """
    Mock function to extract hysteresis loops from Q-T data based on events.
    
    In a real implementation, this would:
    - Extract discharge and concentration data for each event
    - Normalize the data
    - Create hysteresis loop representations
    
    Args:
        qt_data: DataFrame with datetime index and columns ['Qcms', 'turb']
        events: DataFrame with columns ['start', 'end']
    
    Returns:
        List of numpy arrays, each representing a hysteresis loop
    """
    loops = []
    
    for _, event in events.iterrows():
        # Extract data for this event
        mask = (qt_data.index >= event['start']) & (qt_data.index <= event['end'])
        event_data = qt_data[mask]
        
        if len(event_data) > 0:
            # Mock: Create a simple loop representation (normalized Q and C values)
            q_values = event_data['Qcms'].values
            c_values = event_data['turb'].values
            
            # Normalize (mock normalization)
            q_norm = (q_values - q_values.min()) / (q_values.max() - q_values.min() + 1e-10)
            c_norm = (c_values - c_values.min()) / (c_values.max() - c_values.min() + 1e-10)
            
            # Create loop representation
            loop = np.column_stack([q_norm, c_norm])
            loops.append(loop)
    
    return loops


def get_bmu_for_loop(loop: np.ndarray, som_shape: Tuple[int, int] = (8, 8)) -> Tuple[int, int]:
    """
    Mock function to get the Best Matching Unit (BMU) for a hysteresis loop.
    
    In a real implementation, this would:
    - Use the trained SOM model
    - Calculate distances to all neurons
    - Return the coordinates of the closest neuron
    
    Args:
        loop: Numpy array representing the hysteresis loop
        som_shape: Shape of the SOM grid (rows, cols)
    
    Returns:
        Tuple of (row, col) coordinates of the BMU
    """
    # Mock: Generate pseudo-random but deterministic BMU based on loop characteristics
    if len(loop) == 0:
        return (0, 0)
    
    # Use loop statistics to generate a "realistic" BMU
    mean_q = np.mean(loop[:, 0]) if loop.shape[1] > 0 else 0.5
    mean_c = np.mean(loop[:, 1]) if loop.shape[1] > 1 else 0.5
    
    # Map to SOM coordinates
    row = int(mean_c * (som_shape[0] - 1))
    col = int(mean_q * (som_shape[1] - 1))
    
    # Add some variation based on loop shape
    variation = int(np.std(loop) * 2) % 2
    row = min(max(0, row + variation), som_shape[0] - 1)
    col = min(max(0, col + variation), som_shape[1] - 1)
    
    return (row, col)


def calculate_loop_distance(loop: np.ndarray, bmu: Tuple[int, int]) -> float:
    """
    Mock function to calculate distance between a loop and its BMU.
    
    In a real implementation, this would:
    - Calculate the distance between the loop and the SOM prototype at the BMU
    - Use appropriate distance metric (e.g., Euclidean, DTW)
    
    Args:
        loop: Numpy array representing the hysteresis loop
        bmu: Tuple of (row, col) coordinates of the BMU
    
    Returns:
        Float representing the distance (lower is better match)
    """
    if len(loop) == 0:
        return 0.0
    
    # Mock: Generate realistic distance based on loop characteristics and BMU position
    # Distance should be lower for loops that are "typical" for their BMU
    mean_q = np.mean(loop[:, 0]) if loop.shape[1] > 0 else 0.5
    mean_c = np.mean(loop[:, 1]) if loop.shape[1] > 1 else 0.5
    std_combined = np.std(loop)
    
    # Expected position based on BMU
    expected_c = bmu[0] / 7.0  # Normalize to 0-1
    expected_q = bmu[1] / 7.0
    
    # Calculate mock distance
    position_diff = np.sqrt((mean_q - expected_q)**2 + (mean_c - expected_c)**2)
    variability_factor = std_combined * 0.5
    
    # Combine factors to get distance (0.0 to ~2.0 range)
    distance = position_diff + variability_factor
    
    return round(distance, 4)


def calculate_bmu_for_events(
    qt_data: pd.DataFrame, 
    events: pd.DataFrame,
    som_shape: Tuple[int, int] = (8, 8)
) -> pd.DataFrame:
    """
    Calculate BMU coordinates for all events.
    
    Args:
        qt_data: DataFrame with datetime index and columns ['Qcms', 'turb']
        events: DataFrame with columns ['start', 'end']
        som_shape: Shape of the SOM grid
    
    Returns:
        DataFrame with event info, BMU coordinates, and distances
    """
    # Extract loops
    loops = extract_hysteresis_loops(qt_data, events)
    
    # Calculate BMU for each loop
    bmus = [get_bmu_for_loop(loop, som_shape) for loop in loops]
    
    # Calculate distances
    distances = [calculate_loop_distance(loop, bmu) for loop, bmu in zip(loops, bmus)]
    
    # Create results dataframe
    results = events.copy()
    results['BMU_Row'] = [bmu[0] for bmu in bmus]
    results['BMU_Col'] = [bmu[1] for bmu in bmus]
    results['Distance'] = distances
    results['Event_ID'] = range(1, len(results) + 1)
    
    # Reorder columns
    results = results[['Event_ID', 'start', 'end', 'BMU_Row', 'BMU_Col', 'Distance']]
    
    return results


def create_frequency_map(
    bmu_results: pd.DataFrame,
    som_shape: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Create a frequency distribution map showing how many loops map to each BMU.
    
    Args:
        bmu_results: DataFrame with columns ['BMU_Row', 'BMU_Col']
        som_shape: Shape of the SOM grid
    
    Returns:
        2D numpy array with frequency counts
    """
    # Initialize frequency map
    freq_map = np.zeros(som_shape)
    
    # Count occurrences at each BMU
    for _, row in bmu_results.iterrows():
        bmu_row = int(row['BMU_Row'])
        bmu_col = int(row['BMU_Col'])
        freq_map[bmu_row, bmu_col] += 1
    
    return freq_map


def plot_frequency_map(
    freq_map: np.ndarray,
    title: str = "Frequency Distribution of Hysteresis Loops"
) -> plt.Figure:
    """
    Create a matplotlib heatmap of the frequency distribution.
    
    Args:
        freq_map: 2D numpy array with frequency counts
        title: Plot title
    
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(freq_map, cmap='YlOrRd', aspect='auto', origin='upper')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Events', rotation=270, labelpad=20)
    
    # Set ticks and labels
    rows, cols = freq_map.shape
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels(np.arange(cols))
    ax.set_yticklabels(np.arange(rows))
    
    # Add grid
    ax.set_xticks(np.arange(cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(rows + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    
    # Add text annotations
    for i in range(rows):
        for j in range(cols):
            count = int(freq_map[i, j])
            if count > 0:
                text = ax.text(j, i, str(count),
                             ha="center", va="center", color="black", fontsize=12)
    
    # Labels and title
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    return fig


def get_loops_for_bmu(
    qt_data: pd.DataFrame,
    events: pd.DataFrame,
    bmu_results: pd.DataFrame,
    target_row: int,
    target_col: int
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Get all hysteresis loops that map to a specific BMU.
    
    Args:
        qt_data: DataFrame with datetime index and columns ['Qcms', 'turb']
        events: DataFrame with columns ['start', 'end']
        bmu_results: DataFrame with BMU classifications
        target_row: Target BMU row coordinate
        target_col: Target BMU column coordinate
    
    Returns:
        Tuple of (list of loop arrays, list of event IDs)
    """
    # Find events that map to this BMU
    matching_events = bmu_results[
        (bmu_results['BMU_Row'] == target_row) & 
        (bmu_results['BMU_Col'] == target_col)
    ]
    
    if len(matching_events) == 0:
        return [], []
    
    # Extract loops for these events
    loops = []
    event_ids = []
    
    for _, event_row in matching_events.iterrows():
        event_id = event_row['Event_ID']
        start = event_row['start']
        end = event_row['end']
        
        # Extract data for this event
        mask = (qt_data.index >= start) & (qt_data.index <= end)
        event_data = qt_data[mask]
        
        if len(event_data) > 0:
            q_values = event_data['Qcms'].values
            c_values = event_data['turb'].values
            
            # Store raw values (not normalized) for visualization
            loop = np.column_stack([q_values, c_values])
            loops.append(loop)
            event_ids.append(event_id)
    
    return loops, event_ids


def get_average_loop_for_bmu(
    qt_data: pd.DataFrame,
    events: pd.DataFrame,
    bmu_results: pd.DataFrame,
    target_row: int,
    target_col: int,
    num_points: int = 50
) -> np.ndarray:
    """
    Create a mock average loop for a specific BMU prototype.
    
    Args:
        qt_data: DataFrame with datetime index and columns ['Qcms', 'turb']
        events: DataFrame with columns ['start', 'end']
        bmu_results: DataFrame with BMU classifications
        target_row: Target BMU row coordinate
        target_col: Target BMU column coordinate
        num_points: Number of points in the averaged loop
    
    Returns:
        Numpy array representing the average loop
    """
    # Get all loops for this BMU
    loops, _ = get_loops_for_bmu(qt_data, events, bmu_results, target_row, target_col)
    
    if len(loops) == 0:
        # Return empty array if no loops
        return np.array([])
    
    # Resample all loops to same number of points
    resampled_loops = []
    for loop in loops:
        if len(loop) < 2:
            continue
        # Simple linear interpolation to num_points
        indices = np.linspace(0, len(loop) - 1, num_points)
        q_interp = np.interp(indices, np.arange(len(loop)), loop[:, 0])
        c_interp = np.interp(indices, np.arange(len(loop)), loop[:, 1])
        resampled_loops.append(np.column_stack([q_interp, c_interp]))
    
    if len(resampled_loops) == 0:
        return np.array([])
    
    # Calculate average
    avg_loop = np.mean(resampled_loops, axis=0)
    
    return avg_loop


def plot_hysteresis_loops(
    loops: List[np.ndarray],
    event_ids: List[int],
    bmu_row: int,
    bmu_col: int
) -> go.Figure:
    """
    Create an interactive Plotly plot of hysteresis loops.
    
    Args:
        loops: List of loop arrays (Q, C pairs)
        event_ids: List of event IDs corresponding to loops
        bmu_row: BMU row coordinate
        bmu_col: BMU column coordinate
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    if len(loops) == 0:
        # Empty plot with message
        fig.add_annotation(
            text="No events map to this BMU",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title=f"Hysteresis Loops for BMU ({bmu_row}, {bmu_col})",
            xaxis_title="Discharge (Qcms)",
            yaxis_title="Concentration (turb)",
            height=600
        )
        return fig
    
    # Color palette for different loops
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    # Plot each loop
    for idx, (loop, event_id) in enumerate(zip(loops, event_ids)):
        color = colors[idx % len(colors)]
        
        # Add the loop trace
        fig.add_trace(go.Scatter(
            x=loop[:, 0],
            y=loop[:, 1],
            mode='lines+markers',
            name=f'Event {event_id}',
            line=dict(color=color, width=2),
            marker=dict(size=4, color=color),
            hovertemplate='<b>Event %{fullData.name}</b><br>' +
                         'Q: %{x:.3f} cms<br>' +
                         'C: %{y:.2f}<br>' +
                         '<extra></extra>'
        ))
        
        # Add start marker
        fig.add_trace(go.Scatter(
            x=[loop[0, 0]],
            y=[loop[0, 1]],
            mode='markers',
            marker=dict(size=10, color=color, symbol='circle', line=dict(color='white', width=2)),
            showlegend=False,
            hovertemplate='<b>Start</b><br>Q: %{x:.3f}<br>C: %{y:.2f}<extra></extra>'
        ))
        
        # Add end marker
        fig.add_trace(go.Scatter(
            x=[loop[-1, 0]],
            y=[loop[-1, 1]],
            mode='markers',
            marker=dict(size=10, color=color, symbol='square', line=dict(color='white', width=2)),
            showlegend=False,
            hovertemplate='<b>End</b><br>Q: %{x:.3f}<br>C: %{y:.2f}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=f"Hysteresis Loops for BMU ({bmu_row}, {bmu_col}) - {len(loops)} Event(s)",
        xaxis_title="Discharge (Qcms)",
        yaxis_title="Concentration (turb)",
        hovermode='closest',
        height=600,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig


def plot_loop_comparison(
    loops: List[np.ndarray],
    labels: List[str],
    title: str = "Loop Comparison"
) -> go.Figure:
    """
    Create an interactive Plotly plot comparing multiple hysteresis loops.
    
    Args:
        loops: List of loop arrays (Q, C pairs)
        labels: List of labels for each loop
        title: Plot title
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    if len(loops) == 0:
        # Empty plot with message
        fig.add_annotation(
            text="Select events from the table to compare loops",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray")
        )
        fig.update_layout(
            title=title,
            xaxis_title="Discharge (Qcms)",
            yaxis_title="Concentration (turb)",
            height=500
        )
        return fig
    
    # Color palette
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    # Plot each loop
    for idx, (loop, label) in enumerate(zip(loops, labels)):
        if len(loop) == 0:
            continue
            
        color = colors[idx % len(colors)]
        
        # Determine line style based on label
        line_style = 'dash' if 'Prototype' in label else 'solid'
        line_width = 3 if 'Prototype' in label else 2
        
        # Add the loop trace
        fig.add_trace(go.Scatter(
            x=loop[:, 0],
            y=loop[:, 1],
            mode='lines+markers',
            name=label,
            line=dict(color=color, width=line_width, dash=line_style),
            marker=dict(size=4, color=color),
            hovertemplate=f'<b>{label}</b><br>' +
                         'Q: %{x:.3f} cms<br>' +
                         'C: %{y:.2f}<br>' +
                         '<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Discharge (Qcms)",
        yaxis_title="Concentration (turb)",
        hovermode='closest',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig


# ==================== UI COMPONENT FUNCTIONS ====================

def create_time_series_plot(
    qt_data: pd.DataFrame,
    events_data: pd.DataFrame
) -> go.Figure:
    """
    Create an interactive time series plot with dual axes and event overlays.
    
    Args:
        qt_data: DataFrame with datetime index and columns ['Qcms', 'turb']
        events_data: DataFrame with columns ['start', 'end']
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Add discharge trace
    fig.add_trace(go.Scatter(
        x=qt_data.index,
        y=qt_data['Qcms'],
        mode='lines',
        name='Discharge (Qcms)',
        line=dict(color='#1f77b4', width=1.5),
        yaxis='y1'
    ))
    
    # Add concentration trace
    fig.add_trace(go.Scatter(
        x=qt_data.index,
        y=qt_data['turb'],
        mode='lines',
        name='Turbidity',
        line=dict(color='#ff7f0e', width=1.5),
        yaxis='y2'
    ))
    
    # Add event rectangles
    for idx, event in events_data.iterrows():
        fig.add_vrect(
            x0=event['start'],
            x1=event['end'],
            fillcolor='green',
            opacity=0.15,
            line_width=0,
            annotation_text=f"E{idx+1}" if idx < 5 else "",
            annotation_position="top left"
        )
    
    # Update layout
    fig.update_layout(
        title="Discharge and Concentration Time Series with Hydrologic Events",
        xaxis_title="Date",
        yaxis=dict(
            title="Discharge (Qcms)",
            title_font=dict(color='#1f77b4'),
            tickfont=dict(color='#1f77b4')
        ),
        yaxis2=dict(
            title="Turbidity",
            title_font=dict(color='#ff7f0e'),
            tickfont=dict(color='#ff7f0e'),
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def load_qt_data_from_file(uploaded_file) -> pd.DataFrame:
    """
    Load Q-T time series data from uploaded CSV file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        DataFrame with datetime index and columns ['Qcms', 'turb']
    
    Raises:
        ValueError: If required columns are missing
    """
    qt_data = pd.read_csv(uploaded_file, index_col="datetime", parse_dates=True)
    
    # Validate required columns
    required_cols = ['Qcms', 'turb']
    missing_cols = [col for col in required_cols if col not in qt_data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
    
    return qt_data


def load_events_data_from_file(uploaded_file) -> pd.DataFrame:
    """
    Load events data from uploaded CSV file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        DataFrame with columns ['start', 'end'] as datetime
    
    Raises:
        ValueError: If required columns are missing
    """
    events_data = pd.read_csv(uploaded_file)
    
    # Validate required columns
    required_cols = ['start', 'end']
    missing_cols = [col for col in required_cols if col not in events_data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
    
    # Convert to datetime
    events_data['start'] = pd.to_datetime(events_data['start'])
    events_data['end'] = pd.to_datetime(events_data['end'])
    
    return events_data


def calculate_dataset_metrics(
    qt_data: pd.DataFrame,
    events_data: pd.DataFrame,
    bmu_results: pd.DataFrame
) -> Dict[str, any]:
    """
    Calculate summary metrics for the dataset.
    
    Args:
        qt_data: DataFrame with datetime index
        events_data: DataFrame with event information
        bmu_results: DataFrame with BMU classifications
    
    Returns:
        Dictionary with metric names and values
    """
    unique_bmus = bmu_results[['BMU_Row', 'BMU_Col']].drop_duplicates()
    date_range = (qt_data.index[-1] - qt_data.index[0]).days
    
    return {
        "Total Events": len(events_data),
        "Time Series Records": f"{len(qt_data):,}",
        "Unique BMU Patterns": len(unique_bmus),
        "Date Range (days)": date_range
    }

