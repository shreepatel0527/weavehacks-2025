"""
Optimized UI components with performance improvements
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import deque
import threading
from functools import lru_cache
import asyncio

class PerformanceMonitor:
    """Monitor UI performance metrics"""
    
    def __init__(self):
        self.render_times = deque(maxlen=100)
        self.update_times = deque(maxlen=100)
        self.frame_times = deque(maxlen=100)
        self.last_frame_time = time.time()
    
    def record_render_time(self, duration: float):
        """Record render time"""
        self.render_times.append(duration)
    
    def record_update_time(self, duration: float):
        """Record update time"""
        self.update_times.append(duration)
    
    def record_frame(self):
        """Record frame timing"""
        now = time.time()
        frame_time = now - self.last_frame_time
        self.frame_times.append(frame_time)
        self.last_frame_time = now
    
    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics"""
        return {
            'avg_render_ms': np.mean(self.render_times) * 1000 if self.render_times else 0,
            'avg_update_ms': np.mean(self.update_times) * 1000 if self.update_times else 0,
            'fps': 1.0 / np.mean(self.frame_times) if self.frame_times else 0,
            'render_count': len(self.render_times),
            'update_count': len(self.update_times)
        }

class DataCache:
    """Efficient data caching with TTL"""
    
    def __init__(self, ttl_seconds: float = 1.0):
        self.cache = {}
        self.timestamps = {}
        self.ttl = ttl_seconds
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        with self._lock:
            if key in self.cache:
                if time.time() - self.timestamps[key] < self.ttl:
                    return self.cache[key]
                else:
                    # Expired
                    del self.cache[key]
                    del self.timestamps[key]
        return None
    
    def set(self, key: str, value: Any):
        """Set cached value"""
        with self._lock:
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def clear_expired(self):
        """Clear expired entries"""
        with self._lock:
            now = time.time()
            expired_keys = [
                k for k, t in self.timestamps.items()
                if now - t >= self.ttl
            ]
            for key in expired_keys:
                del self.cache[key]
                del self.timestamps[key]

class OptimizedDataDisplay:
    """Optimized data display with virtualization"""
    
    def __init__(self, max_rows: int = 100):
        self.max_rows = max_rows
        self.cache = DataCache(ttl_seconds=0.5)
        self.perf_monitor = PerformanceMonitor()
    
    @st.cache_data(ttl=1)
    def prepare_dataframe(self, data: List[Dict]) -> pd.DataFrame:
        """Prepare dataframe with caching"""
        start_time = time.time()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Limit rows for performance
        if len(df) > self.max_rows:
            df = df.tail(self.max_rows)
        
        # Format timestamps if present
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['timestamp'] = df['timestamp'].dt.strftime('%H:%M:%S')
        
        # Round numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].round(2)
        
        self.perf_monitor.record_render_time(time.time() - start_time)
        return df
    
    def display_metrics(self, metrics: Dict[str, float], container=None):
        """Display metrics with minimal redraws"""
        start_time = time.time()
        
        # Use container for targeted updates
        target = container or st
        
        # Create columns
        cols = target.columns(len(metrics))
        
        # Display metrics
        for idx, (key, value) in enumerate(metrics.items()):
            with cols[idx]:
                # Format key
                display_key = key.replace('_', ' ').title()
                
                # Format value
                if isinstance(value, float):
                    if value > 1000:
                        display_value = f"{value/1000:.1f}k"
                    else:
                        display_value = f"{value:.2f}"
                else:
                    display_value = str(value)
                
                # Use metric widget
                st.metric(display_key, display_value)
        
        self.perf_monitor.record_render_time(time.time() - start_time)
    
    def display_table_virtualized(self, data: List[Dict], 
                                 page_size: int = 20,
                                 container=None):
        """Display table with virtualization"""
        start_time = time.time()
        
        target = container or st
        
        # Prepare data
        df = self.prepare_dataframe(data)
        
        # Calculate pages
        total_rows = len(df)
        total_pages = (total_rows + page_size - 1) // page_size
        
        # Page selection
        if total_pages > 1:
            page = target.selectbox(
                "Page",
                range(1, total_pages + 1),
                format_func=lambda x: f"Page {x} of {total_pages}"
            )
        else:
            page = 1
        
        # Display current page
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_rows)
        
        target.dataframe(
            df.iloc[start_idx:end_idx],
            use_container_width=True,
            hide_index=True
        )
        
        self.perf_monitor.record_render_time(time.time() - start_time)

class OptimizedCharts:
    """Optimized chart rendering with caching"""
    
    def __init__(self):
        self.cache = DataCache(ttl_seconds=2.0)
        self.perf_monitor = PerformanceMonitor()
        self.last_data_hash = {}
    
    @lru_cache(maxsize=10)
    def _create_line_trace(self, name: str, color: str) -> dict:
        """Create cached line trace configuration"""
        return {
            'name': name,
            'line': {'color': color, 'width': 2},
            'mode': 'lines',
            'type': 'scatter'
        }
    
    def create_realtime_chart(self, data: Dict[str, List[Dict]], 
                            title: str = "Real-time Data",
                            height: int = 400) -> go.Figure:
        """Create optimized real-time chart"""
        start_time = time.time()
        
        # Check cache
        data_hash = hash(str(data))
        cache_key = f"chart_{title}_{data_hash}"
        
        cached_fig = self.cache.get(cache_key)
        if cached_fig is not None:
            return cached_fig
        
        # Create figure with subplots
        fig = make_subplots(
            rows=len(data),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=list(data.keys())
        )
        
        # Color palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Add traces
        for idx, (param, values) in enumerate(data.items()):
            if not values:
                continue
            
            # Extract data
            timestamps = [v['timestamp'] for v in values[-100:]]  # Limit points
            measurements = [v['value'] for v in values[-100:]]
            
            # Add trace
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=measurements,
                    **self._create_line_trace(param, colors[idx % len(colors)])
                ),
                row=idx + 1,
                col=1
            )
            
            # Update y-axis
            fig.update_yaxes(
                title_text=param.replace('_', ' ').title(),
                row=idx + 1,
                col=1
            )
        
        # Update layout
        fig.update_layout(
            height=height,
            title=title,
            showlegend=False,
            margin=dict(l=50, r=20, t=50, b=50),
            template='plotly_white'
        )
        
        # Update x-axis
        fig.update_xaxes(title_text="Time", row=len(data), col=1)
        
        # Cache result
        self.cache.set(cache_key, fig)
        
        self.perf_monitor.record_render_time(time.time() - start_time)
        return fig
    
    def create_gauge_chart(self, value: float, min_val: float, max_val: float,
                          title: str, thresholds: Optional[Dict] = None) -> go.Figure:
        """Create optimized gauge chart"""
        cache_key = f"gauge_{title}_{value}_{min_val}_{max_val}"
        
        cached_fig = self.cache.get(cache_key)
        if cached_fig is not None:
            return cached_fig
        
        # Create gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            gauge={
                'axis': {'range': [min_val, max_val]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [min_val, (min_val + max_val) / 2], 'color': "lightgray"},
                    {'range': [(min_val + max_val) / 2, max_val], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': thresholds.get('critical', max_val * 0.9) if thresholds else max_val * 0.9
                }
            }
        ))
        
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        
        # Cache result
        self.cache.set(cache_key, fig)
        
        return fig

class OptimizedUI:
    """Main optimized UI class"""
    
    def __init__(self):
        self.data_display = OptimizedDataDisplay()
        self.charts = OptimizedCharts()
        self.perf_monitor = PerformanceMonitor()
        self.update_counter = 0
        
        # Rate limiting
        self.last_update_times = {}
        self.min_update_interval = 0.1  # 100ms minimum between updates
    
    def should_update(self, component: str) -> bool:
        """Check if component should update (rate limiting)"""
        now = time.time()
        
        if component not in self.last_update_times:
            self.last_update_times[component] = now
            return True
        
        if now - self.last_update_times[component] >= self.min_update_interval:
            self.last_update_times[component] = now
            return True
        
        return False
    
    def render_safety_dashboard(self, safety_data: Dict, container=None):
        """Render optimized safety dashboard"""
        if not self.should_update('safety_dashboard'):
            return
        
        start_time = time.time()
        target = container or st
        
        # Create columns for gauges
        cols = target.columns(3)
        
        # Temperature gauge
        with cols[0]:
            if 'temperature' in safety_data:
                fig = self.charts.create_gauge_chart(
                    safety_data['temperature'],
                    0, 40,
                    "Temperature (°C)",
                    {'critical': 30}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Pressure gauge
        with cols[1]:
            if 'pressure' in safety_data:
                fig = self.charts.create_gauge_chart(
                    safety_data['pressure'],
                    90, 120,
                    "Pressure (kPa)",
                    {'critical': 110}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Stirring gauge
        with cols[2]:
            if 'stirring_rpm' in safety_data:
                fig = self.charts.create_gauge_chart(
                    safety_data['stirring_rpm'],
                    500, 1500,
                    "Stirring (RPM)",
                    {'critical': 1200}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        self.perf_monitor.record_render_time(time.time() - start_time)
    
    def render_data_table(self, data: List[Dict], title: str = "Data",
                         container=None):
        """Render optimized data table"""
        if not self.should_update(f'table_{title}'):
            return
        
        target = container or st
        
        with target.expander(title, expanded=True):
            self.data_display.display_table_virtualized(
                data,
                page_size=20,
                container=st
            )
    
    def render_performance_metrics(self, container=None):
        """Render UI performance metrics"""
        if not self.should_update('performance'):
            return
        
        target = container or st
        
        metrics = self.perf_monitor.get_metrics()
        
        with target.expander("Performance", expanded=False):
            cols = st.columns(4)
            
            with cols[0]:
                st.metric("FPS", f"{metrics['fps']:.1f}")
            
            with cols[1]:
                st.metric("Render Time", f"{metrics['avg_render_ms']:.1f} ms")
            
            with cols[2]:
                st.metric("Update Time", f"{metrics['avg_update_ms']:.1f} ms")
            
            with cols[3]:
                st.metric("Updates", metrics['update_count'])
    
    def batch_update(self, updates: List[Dict]):
        """Batch multiple updates for efficiency"""
        start_time = time.time()
        
        for update in updates:
            component = update.get('component')
            data = update.get('data')
            
            if component == 'safety':
                self.render_safety_dashboard(data)
            elif component == 'table':
                self.render_data_table(
                    data.get('values', []),
                    data.get('title', 'Data')
                )
        
        self.perf_monitor.record_update_time(time.time() - start_time)
        self.update_counter += 1

# Optimized Streamlit app example
def create_optimized_app():
    """Create optimized Streamlit app"""
    st.set_page_config(
        page_title="Optimized Lab Assistant",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for performance
    st.markdown("""
    <style>
    /* Reduce animations for better performance */
    * {
        -webkit-transition: none !important;
        -moz-transition: none !important;
        -o-transition: none !important;
        transition: none !important;
        animation-duration: 0s !important;
    }
    
    /* Optimize rendering */
    .stDataFrame {
        will-change: transform;
    }
    
    /* Reduce reflows */
    .element-container {
        contain: layout style;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize optimized UI
    if 'ui' not in st.session_state:
        st.session_state.ui = OptimizedUI()
    
    ui = st.session_state.ui
    
    # Main layout
    st.title("🧪 Optimized Lab Assistant")
    
    # Create containers for targeted updates
    safety_container = st.container()
    data_container = st.container()
    perf_container = st.container()
    
    # Simulate data updates
    safety_data = {
        'temperature': 23.5 + np.random.normal(0, 0.5),
        'pressure': 101.3 + np.random.normal(0, 1),
        'stirring_rpm': 1000 + np.random.normal(0, 50)
    }
    
    # Render components
    with safety_container:
        ui.render_safety_dashboard(safety_data)
    
    with data_container:
        # Generate sample data
        sample_data = [
            {
                'timestamp': datetime.now() - timedelta(seconds=i),
                'temperature': 23.5 + np.random.normal(0, 0.5),
                'pressure': 101.3 + np.random.normal(0, 1),
                'status': 'Normal'
            }
            for i in range(50)
        ]
        ui.render_data_table(sample_data, "Recent Readings")
    
    with perf_container:
        ui.render_performance_metrics()
    
    # Record frame
    ui.perf_monitor.record_frame()

if __name__ == "__main__":
    create_optimized_app()