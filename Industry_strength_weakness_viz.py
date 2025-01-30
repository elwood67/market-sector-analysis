import streamlit as st
import pandas as pd
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
import os

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

class IndustryAnalyzer:
    def __init__(self):
        # Get the directory containing the script
        self.base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.data_sets_dir = self.base_dir / 'data' / 'datasets'
        
        # Color schemes
        self.sector_colors = {
            'Technology': '#FF0000',
            'Healthcare': '#90EE90',
            'Industrials': '#FFA500',
            'Consumer Cyclical': '#808080',
            'Utilities': '#800080',
            'Consumer Defensive': '#00CED1',
            'Basic Materials': '#87CEEB',
            'Communication Services': '#FF1493',
            'Financial Services': '#0000FF',
            'Energy': '#32CD32',
            'Real Estate': '#98FB98'
        }
        
        # Create darker shades for weakness visualization
        self.weakness_colors = {k: self.darken_color(v) for k, v in self.sector_colors.items()}

    def darken_color(self, hex_color):
        """Create a darker version of a hex color"""
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
        darkened = tuple(int(c * 0.7) for c in rgb)
        return f'#{darkened[0]:02x}{darkened[1]:02x}{darkened[2]:02x}'

    def load_industry_data(self):
        """Load and process all industry-level data"""
        industry_stats = {}
        sector_stats = {}
        daily_data = []
        
        # Get all JSON files and sort them properly
        json_files = sorted(list(self.data_sets_dir.glob('comparison_date_*.json')))
        
        if not json_files:
            st.error(f"No data files found in {self.data_sets_dir}")
            return {}, [], {}
        
        # Initialize tracking dictionaries
        active_stocks = defaultdict(set)
        industry_history = defaultdict(lambda: {
            'sector': '',
            'total_added': 0,
            'total_removed': 0,
            'max_active': 0,
            'days_tracked': 0,
            'daily_changes': [],
            'volume_history': [],
            'momentum_scores': [],
            'last_momentum_date': None
        })
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                st.error(f"Error reading file {json_file}: {e}")
                continue
            
            # Process date
            parts = json_file.stem.split('_')
            month, day = int(parts[2]), int(parts[3])
            
            current_date = datetime.now()
            year = 2024
            if month > current_date.month or (month == current_date.month and day > current_date.day):
                year = 2023
                
            date = datetime(year, month, day)
            date_str = date.strftime('%Y-%m-%d')
            
            daily_changes = defaultdict(lambda: {
                'added': 0, 
                'removed': 0, 
                'sector': '', 
                'momentum': 0,
                'date': date
            })
            
            for change in data['changes']:
                sector = change.get('sector')
                industry = change.get('industry')
                status = change['status']
                ticker = change['ticker']
                
                if sector != 'Unknown' and industry:
                    industry_history[industry]['sector'] = sector
                    
                    if status == 'added':
                        active_stocks[industry].add(ticker)
                        industry_history[industry]['total_added'] += 1
                        daily_changes[industry]['added'] += 1
                    elif status == 'removed':
                        active_stocks[industry].discard(ticker)
                        industry_history[industry]['total_removed'] += 1
                        daily_changes[industry]['removed'] += 1
                    
                    daily_changes[industry]['sector'] = sector

            # Update industry statistics
            for industry in industry_history.keys():
                active_count = len(active_stocks[industry])
                industry_history[industry]['max_active'] = max(
                    industry_history[industry]['max_active'], 
                    active_count
                )
                industry_history[industry]['days_tracked'] += 1
                
                # Calculate net change
                net_change = daily_changes[industry]['added'] - daily_changes[industry]['removed']
                industry_history[industry]['daily_changes'].append(net_change)
                
                # Improved momentum calculation with gap handling
                last_date = industry_history[industry]['last_momentum_date']
                if last_date:
                    days_gap = (date - last_date).days
                    if days_gap > 1:
                        # Fill gaps with zero momentum
                        for _ in range(days_gap - 1):
                            industry_history[industry]['momentum_scores'].append(0)
                
                # Calculate 5-day momentum with weighted average
                recent_changes = industry_history[industry]['daily_changes'][-5:]
                if len(recent_changes) > 0:
                    # Weight more recent changes higher
                    weights = np.exp(np.linspace(0, 1, len(recent_changes)))
                    momentum = np.average(recent_changes, weights=weights)
                    industry_history[industry]['momentum_scores'].append(momentum)
                    daily_changes[industry]['momentum'] = momentum
                
                industry_history[industry]['last_momentum_date'] = date
                
                # Track trading volume
                daily_volume = daily_changes[industry]['added'] + daily_changes[industry]['removed']
                industry_history[industry]['volume_history'].append(daily_volume)
            
            daily_data.append({
                'date': date_str,
                'changes': dict(daily_changes),
                'active_counts': {ind: len(stocks) for ind, stocks in active_stocks.items()}
            })
        
        # Calculate final metrics
        metrics = {}
        for industry, history in industry_history.items():
            if history['days_tracked'] > 0:
                metrics[industry] = self.calculate_industry_metrics(industry, history, daily_data)
        
        return metrics, daily_data, industry_history

    def calculate_industry_metrics(self, industry, history, daily_data):
        """Calculate comprehensive metrics for an industry"""
        # Basic volume metrics
        total_volume = history['total_added'] + history['total_removed']
        avg_active = sum(d['active_counts'].get(industry, 0) for d in daily_data[-7:]) / 7
        
        # Calculate volatility and trends
        changes = history['daily_changes']
        volatility = np.std(changes) if changes else 0
        recent_trend = sum(changes[-7:]) if len(changes) >= 7 else 0
        
        # Calculate momentum with exponential moving average
        momentum_scores = history['momentum_scores']
        if len(momentum_scores) >= 5:
            weights = np.exp(np.linspace(0, 1, 5))
            current_momentum = np.average(momentum_scores[-5:], weights=weights)
            momentum_change = current_momentum - momentum_scores[-5]
        else:
            current_momentum = momentum_scores[-1] if momentum_scores else 0
            momentum_change = 0
        
        # Calculate volume trend
        volume_history = history['volume_history']
        recent_volume = sum(volume_history[-5:]) / 5 if volume_history else 0
        
        # Volatility factor (reduces both scores)
        volatility_factor = max(0, 1 - (volatility * 0.15))
        
        # Strength Score Components
        addition_ratio = history['total_added'] / max(total_volume, 1)
        current_strength = avg_active / max(history['max_active'], 1)
        trend_strength = max(recent_trend, 0) / max(history['max_active'], 1)
        
        raw_strength_score = (
            (addition_ratio * 0.3) +
            (current_strength * 0.3) +
            (trend_strength * 0.4)
        )
        strength_score = raw_strength_score * volatility_factor
        
        # Weakness Score Components
        removal_ratio = history['total_removed'] / max(total_volume, 1)
        current_weakness = 1 - current_strength
        trend_weakness = abs(min(recent_trend, 0)) / max(history['max_active'], 1)
        
        raw_weakness_score = (
            (removal_ratio * 0.3) +
            (current_weakness * 0.3) +
            (trend_weakness * 0.4)
        )
        weakness_score = raw_weakness_score * volatility_factor
        
        return {
            'sector': history['sector'],
            'strength_score': round(strength_score * 100, 2),
            'weakness_score': round(weakness_score * 100, 2),
            'active_stocks': daily_data[-1]['active_counts'].get(industry, 0),
            'total_added': history['total_added'],
            'total_removed': history['total_removed'],
            'recent_trend': recent_trend,
            'volatility': round(volatility, 2),
            'momentum': round(current_momentum, 2),
            'momentum_change': round(momentum_change, 2),
            'volume_trend': round(recent_volume, 2),
            'max_active': history['max_active'],
            'raw_strength': round(raw_strength_score * 100, 2),
            'raw_weakness': round(raw_weakness_score * 100, 2),
            'volatility_factor': round(volatility_factor, 2)
        }

def main():
    st.set_page_config(page_title="Industry Analysis", layout="wide")
    
    if not check_password():
        st.stop()
    
    analyzer = IndustryAnalyzer()
    
    try:
        # Load and analyze data
        metrics, daily_data, industry_history = analyzer.load_industry_data()
        
        if not metrics:
            st.error("No data loaded. Please check your data directory.")
            st.stop()
        
        # Create tabs for different views
        tabs = st.tabs(["Strong Industries", "Weak Industries", "Momentum Analysis", "Comparison"])
        
        with tabs[0]:
            st.markdown("""
            ### Strength Score Components:
            - 30% Addition Ratio (Added vs Total volume)
            - 30% Current Strength (Active vs Historical max)
            - 40% Recent Trend (Positive changes in last 7 days)
            - Both Strength and Weakness scores are reduced by volatility
            """)
            st.plotly_chart(analyzer.create_analysis_chart(metrics, 'strength'), use_container_width=True)
            st.plotly_chart(analyzer.create_momentum_chart(metrics, 'strength'), use_container_width=True)
        
        with tabs[1]:
            st.markdown("""
            ### Weakness Score Components:
            - 30% Removal Ratio (Removed vs Total volume)
            - 30% Current Weakness (Below historical max)
            - 40% Recent Trend (Negative changes in last 7 days)
            - Both Strength and Weakness scores are reduced by volatility
            """)
            st.plotly_chart(analyzer.create_analysis_chart(metrics, 'weakness'), use_container_width=True)
            st.plotly_chart(analyzer.create_momentum_chart(metrics, 'weakness'), use_container_width=True)
        
        with tabs[2]:
            st.markdown("""
            ### Momentum Analysis
            - Bar height shows current momentum
            - Line shows momentum change (acceleration)
            - Colors indicate sectors
            - Higher values indicate stronger trends
            """)
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(analyzer.create_volatility_chart(metrics), use_container_width=True)
            
            # Detailed metrics table
            with col2:
                df = pd.DataFrame([
                    {
                        'Industry': industry,
                        'Sector': m['sector'],
                        'Active Stocks': m['active_stocks'],
                        'Strength': m['strength_score'],
                        'Weakness': m['weakness_score'],
                        'Momentum': m['momentum'],
                        'Volatility': m['volatility']
                    }
                    for industry, m in metrics.items()
                ])
                st.dataframe(df.sort_values('Momentum', ascending=False))
        
        with tabs[3]:
            st.markdown("""
            ### Strength vs Weakness Matrix
            - Higher strength and lower weakness scores indicate bullish trends
            - Higher weakness and lower strength scores indicate bearish trends
            - Color indicates sector
            """)
            st.plotly_chart(analyzer.create_score_comparison(metrics), use_container_width=True)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()