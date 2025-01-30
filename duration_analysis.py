import streamlit as st
import pandas as pd
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
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

class EnhancedSectorVisualizer:
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

    def load_all_data(self):
        """Load and process all comparison data from JSON files"""
        stock_tracking = {}
        sector_changes = {}
        daily_data = []
        
        # Get all JSON files and sort them properly by date
        json_files = sorted(list(self.data_sets_dir.glob('comparison_date_*.json')))
        
        if not json_files:
            st.error(f"No data files found in {self.data_sets_dir}")
            return {}, [], {}
        
        active_stocks = {}
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                st.error(f"Error reading file {json_file}: {e}")
                continue

            # Extract date parts from filename (format: comparison_date_MM_DD.json)
            parts = json_file.stem.split('_')
            month, day = int(parts[2]), int(parts[3])
            
            # Handle year transition - if month/day is after current date, it's from previous year
            current_date = datetime.now()
            year = 2024
            if month > current_date.month or (month == current_date.month and day > current_date.day):
                year = 2023
                
            date = datetime(year, month, day)
            date_str = date.strftime('%Y-%m-%d')
            
            # Process daily changes
            sector_counts = {}
            industry_counts = {}
            
            for change in data['changes']:
                ticker = change['ticker']
                sector = change.get('sector')
                industry = change.get('industry')
                status = change['status']
                
                if sector != 'Unknown':
                    # Track individual stock history
                    if status == 'added':
                        stock_tracking[ticker] = {
                            'start_date': date_str,
                            'sector': sector,
                            'industry': industry,
                            'status': 'active'
                        }
                        if sector not in active_stocks:
                            active_stocks[sector] = set()
                        active_stocks[sector].add(ticker)
                        
                    elif status == 'removed':
                        if ticker in stock_tracking and stock_tracking[ticker]['status'] == 'active':
                            stock_tracking[ticker].update({
                                'end_date': date_str,
                                'status': 'removed',
                                'duration': (date - datetime.strptime(stock_tracking[ticker]['start_date'], '%Y-%m-%d')).days
                            })
                            if sector in active_stocks:
                                active_stocks[sector].discard(ticker)
                    
                    # Update daily counts
                    if sector not in sector_counts:
                        sector_counts[sector] = 0
                    if status == 'added':
                        sector_counts[sector] += 1
                    elif status == 'removed':
                        sector_counts[sector] -= 1
                        
                    # Update industry counts
                    if sector not in industry_counts:
                        industry_counts[sector] = {}
                    if industry not in industry_counts[sector]:
                        industry_counts[sector][industry] = 0
                    if status == 'added':
                        industry_counts[sector][industry] += 1
                    elif status == 'removed':
                        industry_counts[sector][industry] -= 1
            
            # Calculate active stock counts for this date
            active_counts = {sector: len(stocks) for sector, stocks in active_stocks.items()}
            
            daily_data.append({
                'date': date_str,
                'sector_changes': sector_counts,
                'industry_changes': industry_counts,
                'active_counts': active_counts
            })
            
            # Store final industry changes
            if json_file == json_files[-1]:
                sector_changes = industry_counts
        
        # Sort daily data by date
        daily_data.sort(key=lambda x: x['date'])
        return daily_data, sector_changes, stock_tracking

    def analyze_durations(self, stock_tracking):
        """Analyze how long stocks maintain bullish structure"""
        sector_stats = {}
        industry_stats = {}
        
        # Calculate statistics by sector and industry
        for ticker, data in stock_tracking.items():
            if 'duration' in data:  # Only process completed cycles
                sector = data['sector']
                industry = data['industry']
                duration = data['duration']
                
                # Update sector stats
                if sector not in sector_stats:
                    sector_stats[sector] = []
                sector_stats[sector].append(duration)
                
                # Update industry stats
                if sector not in industry_stats:
                    industry_stats[sector] = {}
                if industry not in industry_stats[sector]:
                    industry_stats[sector][industry] = []
                industry_stats[sector][industry].append(duration)
        
        # Calculate summary statistics
        sector_summary = {}
        industry_summary = {}
        
        for sector, durations in sector_stats.items():
            if durations:  # Only process sectors with data
                sector_summary[sector] = {
                    'avg_duration': round(sum(durations) / len(durations), 1),
                    'median_duration': sorted(durations)[len(durations)//2],
                    'max_duration': max(durations),
                    'min_duration': min(durations),
                    'total_stocks': len(durations)
                }
        
        for sector, industries in industry_stats.items():
            industry_summary[sector] = {}
            for industry, durations in industries.items():
                if len(durations) > 0:  # Only process industries with data
                    industry_summary[sector][industry] = {
                        'avg_duration': round(sum(durations) / len(durations), 1),
                        'median_duration': sorted(durations)[len(durations)//2],
                        'max_duration': max(durations),
                        'min_duration': min(durations),
                        'total_stocks': len(durations)
                    }
        
        return sector_summary, industry_summary

    def create_trend_chart(self, data):
        """Create sector trends line chart using Plotly"""
        dates = [d['date'] for d in data]
        sectors = set()
        for d in data:
            sectors.update(d['sector_changes'].keys())
        
        fig = go.Figure()
        
        for sector in sectors:
            values = [d['sector_changes'].get(sector, 0) for d in data]
            fig.add_trace(go.Scatter(
                x=dates,
                y=values,
                name=sector,
                line=dict(color=self.sector_colors.get(sector, '#999999')),
                mode='lines'
            ))
        
        fig.update_layout(
            title='Sector Changes Over Time',
            xaxis_title='Date',
            yaxis_title='Net Change',
            hovermode='x unified',
            template='plotly_dark',
            height=600,
            xaxis=dict(
                type='date',
                tickformat='%Y-%m-%d'
            )
        )
        
        return fig

    def create_duration_chart(self, sector_summary):
        """Create duration analysis chart"""
        sectors = list(sector_summary.keys())
        avg_durations = [sector_summary[s]['avg_duration'] for s in sectors]
        median_durations = [sector_summary[s]['median_duration'] for s in sectors]
        
        # Sort sectors by average duration
        sorted_indices = sorted(range(len(avg_durations)), key=lambda k: avg_durations[k], reverse=True)
        sectors = [sectors[i] for i in sorted_indices]
        avg_durations = [avg_durations[i] for i in sorted_indices]
        median_durations = [median_durations[i] for i in sorted_indices]
        
        fig = go.Figure()
        
        # Add average duration bars
        fig.add_trace(go.Bar(
            name='Average Duration',
            x=sectors,
            y=avg_durations,
            marker_color='#4CAF50'
        ))
        
        # Add median duration bars
        fig.add_trace(go.Bar(
            name='Median Duration',
            x=sectors,
            y=median_durations,
            marker_color='#2196F3'
        ))
        
        fig.update_layout(
            title='Bullish Structure Duration by Sector',
            xaxis_title='Sector',
            yaxis_title='Days',
            barmode='group',
            template='plotly_dark',
            height=500,
            xaxis={'tickangle': -45}
        )
        
        return fig

    def create_industry_duration_chart(self, industry_summary):
        """Create industry-level duration analysis charts"""
        figs = []
        
        for sector, industries in industry_summary.items():
            if industries:  # Only process sectors with data
                # Sort industries by average duration
                sorted_industries = sorted(
                    industries.items(),
                    key=lambda x: x[1]['avg_duration'],
                    reverse=True
                )[:5]  # Top 5 industries
                
                industry_names = [x[0] for x in sorted_industries]
                avg_durations = [x[1]['avg_duration'] for x in sorted_industries]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=industry_names,
                    y=avg_durations,
                    marker_color=self.sector_colors.get(sector, '#999999'),
                    text=[f"{x:.1f}" for x in avg_durations],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title=f'{sector} - Top Industries by Duration',
                    xaxis_title='Industry',
                    yaxis_title='Average Days',
                    template='plotly_dark',
                    height=300,
                    margin=dict(b=100),  # Add more bottom margin for long labels
                    xaxis={'tickangle': -45}
                )
                
                figs.append(fig)
        
        return figs

def main():
    st.set_page_config(page_title="Duration Analysis", layout="wide")
    
    if not check_password():
        st.stop()
    
    visualizer = EnhancedSectorVisualizer()
    
    try:
        # Load all data
        daily_data, sector_changes, stock_tracking = visualizer.load_all_data()
        
        if not daily_data:
            st.error("No data was loaded. Please check your data directory.")
            st.stop()
            
        sector_summary, industry_summary = visualizer.analyze_durations(stock_tracking)
        
        # Display trend chart
        st.title("Enhanced Sector Analysis Dashboard")
        st.plotly_chart(visualizer.create_trend_chart(daily_data), use_container_width=True)
        
        # Display duration analysis
        st.title("Bullish Structure Duration Analysis")
        st.plotly_chart(visualizer.create_duration_chart(sector_summary), use_container_width=True)
        
        # Display sector statistics
        st.title("Sector Statistics")
        cols = st.columns(3)
        sorted_sectors = dict(sorted(sector_summary.items(), 
                                   key=lambda x: x[1]['avg_duration'], 
                                   reverse=True))
        
        for i, (sector, stats) in enumerate(sorted_sectors.items()):
            with cols[i % 3]:
                st.metric(
                    label=sector,
                    value=f"{stats['avg_duration']} days avg",
                    delta=f"{stats['total_stocks']} stocks"
                )
        
        # Display industry duration analysis
        st.title("Industry Duration Analysis")
        industry_charts = visualizer.create_industry_duration_chart(industry_summary)
        
        if industry_charts:
            cols = st.columns(2)
            for idx, fig in enumerate(industry_charts):
                with cols[idx % 2]:
                    st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()