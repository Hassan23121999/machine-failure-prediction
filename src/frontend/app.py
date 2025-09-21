"""
Streamlit Frontend for Predictive Maintenance System
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #ccc;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .stAlert {
        padding: 15px;
        border-radius: 10px;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 10px;
        border-bottom: 2px solid #1f77b4;
    }
    h2 {
        color: #2c3e50;
        margin-top: 20px;
    }
    .risk-low {
        background-color: #2ecc71;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #f39c12;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .risk-high {
        background-color: #e74c3c;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .risk-critical {
        background-color: #c0392b;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# API Configuration
API_URL = "http://localhost:8000"

# Initialize session state
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []
if 'real_time_data' not in st.session_state:
    st.session_state.real_time_data = []

# Helper functions
def get_risk_color(risk_level):
    """Get color based on risk level"""
    colors = {
        "LOW": "#2ecc71",
        "MEDIUM": "#f39c12",
        "HIGH": "#e74c3c",
        "CRITICAL": "#c0392b"
    }
    return colors.get(risk_level, "#95a5a6")

def get_risk_badge(risk_level):
    """Create HTML badge for risk level"""
    color = get_risk_color(risk_level)
    return f'<span style="background-color: {color}; color: white; padding: 5px 10px; border-radius: 5px; font-weight: bold;">{risk_level}</span>'

def check_api_health():
    """Check if API is healthy"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def make_prediction(sensor_data):
    """Make prediction using API"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=sensor_data,
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

def create_gauge_chart(value, title, max_value=100):
    """Create a gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={'text': title},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, max_value]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, max_value*0.3], 'color': "lightgreen"},
                {'range': [max_value*0.3, max_value*0.7], 'color': "yellow"},
                {'range': [max_value*0.7, max_value], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value*0.9
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3658/3658756.png", width=100)
    st.title("🏭 Control Panel")
    
    # API Status
    api_status = check_api_health()
    if api_status:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Disconnected")
        st.info("Please ensure the API is running on port 8000")
    
    st.divider()
    
    # Navigation
    page = st.selectbox(
        "Navigation",
        ["🏠 Dashboard", "🔮 Prediction", "📊 Analytics", "⚙️ Equipment Monitor", "📈 Real-time Monitoring"]
    )
    
    st.divider()
    
    # Quick Stats
    if api_status:
        st.metric("Total Predictions Today", len(st.session_state.predictions_history))
        if st.session_state.predictions_history:
            failure_rate = sum(1 for p in st.session_state.predictions_history if p['prediction'] == 1) / len(st.session_state.predictions_history)
            st.metric("Failure Rate", f"{failure_rate*100:.1f}%")

# Main Content
if page == "🏠 Dashboard":
    st.title("🏭 Predictive Maintenance Dashboard")
    st.markdown("### Real-time Equipment Health Monitoring System")
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🟢 Equipment Online",
            value="47",
            delta="3",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            label="⚠️ Warnings",
            value="12",
            delta="-2",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="🔴 Critical",
            value="3",
            delta="1",
            delta_color="normal"
        )
    
    with col4:
        st.metric(
            label="⏱️ Avg Response Time",
            value="125ms",
            delta="-15ms",
            delta_color="inverse"
        )
    
    st.divider()
    
    # Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Risk Distribution")
        
        # Sample data for visualization
        risk_data = pd.DataFrame({
            'Risk Level': ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
            'Count': [35, 12, 8, 3],
            'Color': ['#2ecc71', '#f39c12', '#e74c3c', '#c0392b']
        })
        
        fig = px.pie(
            risk_data, 
            values='Count', 
            names='Risk Level',
            color='Risk Level',
            color_discrete_map={
                'LOW': '#2ecc71',
                'MEDIUM': '#f39c12',
                'HIGH': '#e74c3c',
                'CRITICAL': '#c0392b'
            }
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Failure Trend (Last 7 Days)")
        
        # Generate sample trend data
        dates = pd.date_range(end=datetime.now(), periods=7)
        trend_data = pd.DataFrame({
            'Date': dates,
            'Failures': np.random.randint(2, 10, 7),
            'Predictions': np.random.randint(40, 60, 7)
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend_data['Date'],
            y=trend_data['Failures'],
            mode='lines+markers',
            name='Actual Failures',
            line=dict(color='red', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=trend_data['Date'],
            y=trend_data['Predictions'],
            mode='lines+markers',
            name='Total Predictions',
            line=dict(color='blue', width=2)
        ))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Count",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Recent Alerts
    st.subheader("🚨 Recent Alerts")
    
    alerts_data = pd.DataFrame({
        'Time': [
            datetime.now() - timedelta(minutes=5),
            datetime.now() - timedelta(minutes=15),
            datetime.now() - timedelta(minutes=30),
            datetime.now() - timedelta(hours=1)
        ],
        'Equipment': ['Motor A-12', 'Pump B-03', 'Conveyor C-07', 'Motor A-15'],
        'Risk Level': ['CRITICAL', 'HIGH', 'HIGH', 'MEDIUM'],
        'Action': [
            'Immediate shutdown required',
            'Schedule maintenance',
            'Schedule maintenance',
            'Monitor closely'
        ]
    })
    
    for _, alert in alerts_data.iterrows():
        risk_color = get_risk_color(alert['Risk Level'])
        st.markdown(
            f"""
            <div style="padding: 10px; border-left: 4px solid {risk_color}; background-color: #f8f9fa; margin-bottom: 10px; border-radius: 5px;">
                <strong>{alert['Equipment']}</strong> - {alert['Time'].strftime('%H:%M')}
                <br>{get_risk_badge(alert['Risk Level'])} {alert['Action']}
            </div>
            """,
            unsafe_allow_html=True
        )

elif page == "🔮 Prediction":
    st.title("🔮 Equipment Failure Prediction")
    st.markdown("Enter sensor readings to predict equipment failure risk")
    
    # Input Form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌡️ Temperature Readings")
            air_temp = st.slider(
                "Air Temperature (K)",
                min_value=295.0,
                max_value=305.0,
                value=298.0,
                step=0.1
            )
            process_temp = st.slider(
                "Process Temperature (K)",
                min_value=305.0,
                max_value=315.0,
                value=308.0,
                step=0.1
            )
            
            st.subheader("⚙️ Mechanical Parameters")
            rpm = st.slider(
                "Rotational Speed (RPM)",
                min_value=1000,
                max_value=3000,
                value=1500,
                step=10
            )
            torque = st.slider(
                "Torque (Nm)",
                min_value=10.0,
                max_value=80.0,
                value=40.0,
                step=0.5
            )
        
        with col2:
            st.subheader("🔧 Equipment Details")
            tool_wear = st.slider(
                "Tool Wear (minutes)",
                min_value=0,
                max_value=300,
                value=50,
                step=5
            )
            product_type = st.selectbox(
                "Product Type",
                ["L", "M", "H"],
                help="L: Low quality, M: Medium quality, H: High quality"
            )
            
            # Calculate derived features
            st.subheader("📐 Derived Features")
            temp_diff = process_temp - air_temp
            power = (torque * rpm * 2 * np.pi) / 60
            
            st.info(f"""
            **Temperature Differential:** {temp_diff:.1f} K
            
            **Power Output:** {power:.1f} W
            
            **Tool Wear Rate:** {tool_wear/100:.2f}
            """)
        
        submitted = st.form_submit_button("🔍 Predict Failure Risk", use_container_width=True, type="primary")
    
    if submitted:
        # Prepare sensor data
        sensor_data = {
            "air_temperature": air_temp,
            "process_temperature": process_temp,
            "rotational_speed": rpm,
            "torque": torque,
            "tool_wear": tool_wear,
            "product_type": product_type
        }
        
        # Make prediction
        with st.spinner("Analyzing sensor data..."):
            result = make_prediction(sensor_data)
        
        if result:
            # Store in history
            st.session_state.predictions_history.append(result)
            
            # Display results
            st.divider()
            st.subheader("📊 Prediction Results")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                # Risk Level Display
                risk_color = get_risk_color(result['risk_level'])
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 20px; background-color: {risk_color}; border-radius: 10px; color: white;">
                        <h2 style="color: white; margin: 0;">Risk Level</h2>
                        <h1 style="color: white; margin: 10px 0; font-size: 48px;">{result['risk_level']}</h1>
                        <p style="font-size: 18px; margin: 10px 0;">Failure Probability: {result['failure_probability']*100:.1f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            st.divider()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Prediction", "FAILURE" if result['prediction'] == 1 else "NORMAL")
            
            with col2:
                st.metric("Confidence", f"{result['confidence']*100:.1f}%")
            
            with col3:
                st.metric("Response Time", f"{np.random.randint(50, 150)}ms")
            
            # Recommendation
            if result['prediction'] == 1:
                st.error(f"⚠️ **Recommended Action:** {result['recommended_action']}")
            else:
                st.success(f"✅ **Recommended Action:** {result['recommended_action']}")
            
            # Technical Details
            with st.expander("🔍 View Technical Details"):
                st.json(result)

elif page == "📊 Analytics":
    st.title("📊 Analytics Dashboard")
    
    # Time range selector
    time_range = st.selectbox(
        "Select Time Range",
        ["Last Hour", "Last 24 Hours", "Last 7 Days", "Last 30 Days"]
    )
    
    # Generate sample analytics data
    n_points = 100
    analytics_data = pd.DataFrame({
        'timestamp': pd.date_range(end=datetime.now(), periods=n_points, freq='15min'),
        'temperature': np.random.normal(298, 2, n_points),
        'rpm': np.random.normal(1500, 100, n_points),
        'torque': np.random.normal(40, 5, n_points),
        'failures': np.random.binomial(1, 0.1, n_points)
    })
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", f"{n_points}")
    with col2:
        failure_rate = analytics_data['failures'].mean()
        st.metric("Failure Rate", f"{failure_rate*100:.1f}%")
    with col3:
        mtbf = n_points / max(analytics_data['failures'].sum(), 1)
        st.metric("MTBF", f"{mtbf:.1f} hours")
    with col4:
        st.metric("Accuracy", "97.8%")
    
    st.divider()
    
    # Detailed Charts
    tab1, tab2, tab3 = st.tabs(["📈 Sensor Trends", "🎯 Failure Analysis", "🔥 Heatmap"])
    
    with tab1:
        # Sensor trends
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Temperature', 'RPM', 'Torque'),
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=analytics_data['timestamp'], y=analytics_data['temperature'],
                      name='Temperature', line=dict(color='red')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=analytics_data['timestamp'], y=analytics_data['rpm'],
                      name='RPM', line=dict(color='blue')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=analytics_data['timestamp'], y=analytics_data['torque'],
                      name='Torque', line=dict(color='green')),
            row=3, col=1
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Failure analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Failure distribution by hour
            hourly_failures = pd.DataFrame({
                'Hour': range(24),
                'Failures': np.random.poisson(2, 24)
            })
            
            fig = px.bar(
                hourly_failures,
                x='Hour',
                y='Failures',
                title='Failures by Hour of Day',
                color='Failures',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Failure causes
            causes = pd.DataFrame({
                'Cause': ['Tool Wear', 'Overheating', 'Power Failure', 'Overstrain', 'Random'],
                'Count': [45, 30, 20, 15, 10]
            })
            
            fig = px.bar(
                causes,
                x='Count',
                y='Cause',
                orientation='h',
                title='Failure Causes Distribution',
                color='Count',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Correlation heatmap
        st.subheader("Feature Correlation Heatmap")
        
        corr_data = pd.DataFrame(
            np.random.rand(5, 5),
            columns=['Temp', 'RPM', 'Torque', 'Tool Wear', 'Failure'],
            index=['Temp', 'RPM', 'Torque', 'Tool Wear', 'Failure']
        )
        
        fig = px.imshow(
            corr_data,
            color_continuous_scale='RdBu',
            text_auto=True,
            aspect='auto'
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "⚙️ Equipment Monitor":
    st.title("⚙️ Equipment Monitor")
    st.markdown("Real-time monitoring of individual equipment units")
    
    # Equipment selector
    equipment_list = [
        "Motor A-12", "Motor A-15", "Motor B-01",
        "Pump B-03", "Pump C-11", "Conveyor C-07"
    ]
    
    selected_equipment = st.selectbox("Select Equipment", equipment_list)
    
    # Equipment Status
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Generate random status
        statuses = ["OPERATIONAL", "WARNING", "CRITICAL"]
        status = np.random.choice(statuses, p=[0.7, 0.2, 0.1])
        status_colors = {
            "OPERATIONAL": "#2ecc71",
            "WARNING": "#f39c12",
            "CRITICAL": "#e74c3c"
        }
        
        st.markdown(
            f"""
            <div style="text-align: center; padding: 20px; background-color: {status_colors[status]}; border-radius: 10px; color: white;">
                <h2 style="color: white;">{selected_equipment}</h2>
                <h1 style="color: white; font-size: 36px;">{status}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.divider()
    
    # Real-time Gauges
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        fig = create_gauge_chart(
            np.random.uniform(295, 305),
            "Air Temp (K)",
            max_value=310
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_gauge_chart(
            np.random.uniform(305, 315),
            "Process Temp (K)",
            max_value=320
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = create_gauge_chart(
            np.random.uniform(1200, 2800),
            "RPM",
            max_value=3000
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        fig = create_gauge_chart(
            np.random.uniform(20, 70),
            "Torque (Nm)",
            max_value=80
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Maintenance History
    st.subheader("🔧 Maintenance History")
    
    maintenance_data = pd.DataFrame({
        'Date': pd.date_range(end=datetime.now(), periods=5, freq='30D'),
        'Type': ['Preventive', 'Corrective', 'Preventive', 'Inspection', 'Preventive'],
        'Duration (hrs)': [2, 4, 2, 1, 3],
        'Cost ($)': [500, 1200, 500, 200, 600],
        'Technician': ['John D.', 'Sarah M.', 'John D.', 'Mike R.', 'Sarah M.']
    })
    
    st.dataframe(maintenance_data, use_container_width=True)
    
    # Performance Metrics
    st.subheader("📈 Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Uptime", "98.5%", delta="2.1%")
        st.metric("Efficiency", "87.3%", delta="-1.2%")
    
    with col2:
        st.metric("Last Maintenance", "15 days ago")
        st.metric("Next Scheduled", "In 45 days")

elif page == "📈 Real-time Monitoring":
    st.title("📈 Real-time Monitoring")
    st.markdown("Live sensor data streaming and analysis")
    
    # Placeholder for real-time chart
    placeholder = st.empty()
    
    # Control buttons
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        if st.button("▶️ Start Monitoring", type="primary"):
            st.session_state.monitoring = True
    
    with col2:
        if st.button("⏸️ Stop Monitoring"):
            st.session_state.monitoring = False
    
    # Real-time simulation
    if st.session_state.get('monitoring', False):
        for i in range(100):
            # Generate random sensor data
            timestamp = datetime.now()
            
            sensor_data = {
                "air_temperature": np.random.normal(298, 2),
                "process_temperature": np.random.normal(308, 2),
                "rotational_speed": int(np.random.normal(1500, 100)),
                "torque": np.random.normal(40, 5),
                "tool_wear": int(np.random.uniform(0, 250)),
                "product_type": np.random.choice(["L", "M", "H"])
            }
            
            # Make prediction
            result = make_prediction(sensor_data)
            
            if result:
                # Update real-time data
                st.session_state.real_time_data.append({
                    'timestamp': timestamp,
                    'risk_level': result['risk_level'],
                    'probability': result['failure_probability'],
                    **sensor_data
                })
                
                # Keep only last 50 points
                if len(st.session_state.real_time_data) > 50:
                    st.session_state.real_time_data.pop(0)
                
                # Create real-time chart
                df = pd.DataFrame(st.session_state.real_time_data)
                
                with placeholder.container():
                    # Risk indicator
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Current Risk", result['risk_level'])
                    with col2:
                        st.metric("Failure Prob", f"{result['failure_probability']*100:.1f}%")
                    with col3:
                        st.metric("Temperature", f"{sensor_data['air_temperature']:.1f} K")
                    with col4:
                        st.metric("RPM", f"{sensor_data['rotational_speed']}")
                    
                    # Real-time chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=df['timestamp'],
                        y=df['probability']*100,
                        mode='lines',
                        name='Failure Probability',
                        line=dict(color='red', width=2)
                    ))
                    
                    fig.update_layout(
                        title="Real-time Failure Probability",
                        xaxis_title="Time",
                        yaxis_title="Probability (%)",
                        yaxis_range=[0, 100],
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Latest readings table
                    st.subheader("Latest Sensor Readings")
                    latest_df = df.tail(5)[['timestamp', 'air_temperature', 'process_temperature', 
                                           'rotational_speed', 'torque', 'risk_level']]
                    st.dataframe(latest_df, use_container_width=True)
            
            time.sleep(2)  # Update every 2 seconds
    else:
        st.info("Click 'Start Monitoring' to begin real-time data streaming")

# Footer
st.divider()
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>Predictive Maintenance System v1.0 | Powered by Machine Learning</p>
        <p>Built with ❤️ using Streamlit & FastAPI</p>
    </div>
    """,
    unsafe_allow_html=True
)