import streamlit as st
import pandas as pd
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict

class IndustryAnalyzer:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.data_sets_dir = self.base_dir / 'data_sets'
        
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
            with open(json_file, 'r') as f:
                data = json.load(f)
            
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
        
        # Volatility factor (now reduces both scores)
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

    def create_analysis_chart(self, metrics, mode='strength'):
        """Create industry analysis visualization"""
        score_key = f'{mode}_score'
        sorted_industries = sorted(
            metrics.items(),
            key=lambda x: x[1][score_key],
            reverse=True
        )
        
        industries = []
        scores = []
        colors = []
        hover_text = []
        
        color_scheme = self.sector_colors if mode == 'strength' else self.weakness_colors
        
        for industry, m in sorted_industries[:30]:  # Top 30 industries
            industries.append(industry)
            scores.append(m[score_key])
            colors.append(color_scheme.get(m['sector'], '#999999'))
            
            hover_text.append(
                f"Industry: {industry}<br>" +
                f"Sector: {m['sector']}<br>" +
                f"{mode.title()} Score: {m[score_key]}<br>" +
                f"Raw Score: {m[f'raw_{mode}']}<br>" +
                f"Volatility Impact: {m['volatility_factor']}<br>" +
                f"Active Stocks: {m['active_stocks']}<br>" +
                f"Recent Trend: {m['recent_trend']}<br>" +
                f"Momentum: {m['momentum']:.2f}<br>" +
                f"Momentum Change: {m['momentum_change']:.2f}"
            )
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=scores,
            y=industries,
            orientation='h',
            marker_color=colors,
            text=[f"{score:.1f}" for score in scores],
            textposition='outside',
            hovertext=hover_text,
            hoverinfo='text'
        ))
        
        title = 'Industry Strength Analysis' if mode == 'strength' else 'Industry Weakness Analysis'
        fig.update_layout(
            title=title,
            xaxis_title=f'{mode.title()} Score',
            yaxis_title='Industry',
            template='plotly_dark',
            height=800,
            margin=dict(l=200, r=100)
        )
        
        return fig

    def create_momentum_chart(self, metrics, mode='strength'):
        """Create momentum analysis chart"""
        momentum_data = []
        
        score_key = f'{mode}_score'
        for industry, m in metrics.items():
            momentum_data.append({
                'industry': industry,
                'sector': m['sector'],
                'momentum': m['momentum'],
                'momentum_change': m['momentum_change'],
                'score': m[score_key],
                'volume': m['volume_trend']
            })
        
        # Sort by momentum and score
        momentum_data.sort(key=lambda x: (x['momentum'], x['score']), reverse=True)
        momentum_data = momentum_data[:20]  # Top 20
        
        fig = go.Figure()
        
        # Add momentum bars
        fig.add_trace(go.Bar(
            name='Current Momentum',
            x=[d['industry'] for d in momentum_data],
            y=[d['momentum'] for d in momentum_data],
            marker_color=[self.sector_colors.get(d['sector'], '#999999') for d in momentum_data],
            hovertemplate=(
                "<b>%{x}</b><br>" +
                "Momentum: %{y:.2f}<br>" +
                "<extra></extra>"
            )
        ))
        
        # Add momentum change line
        fig.add_trace(go.Scatter(
            name='Momentum Change',
            x=[d['industry'] for d in momentum_data],
            y=[d['momentum_change'] for d in momentum_data],
            mode='lines+markers',
            line=dict(color='white', width=2),
            marker=dict(size=6),
            hovertemplate=(
                "<b>%{x}</b><br>" +
                "Momentum Change: %{y:.2f}<br>" +
                "<extra></extra>"
            )
        ))
        
        fig.update_layout(
            title=f'Industry Momentum Analysis ({mode.title()})',
            xaxis_title='Industry',
            yaxis_title='Momentum Score',
            template='plotly_dark',
            height=600,
            xaxis={'tickangle': -45},
            showlegend=True,
            barmode='relative',
            hovermode='x unified'
        )
        
        return fig

    def create_score_comparison(self, metrics):
        """Create scatter plot comparing strength vs weakness"""
        industries = list(metrics.keys())
        strength_scores = [m['strength_score'] for m in metrics.values()]
        weakness_scores = [m['weakness_score'] for m in metrics.values()]
        sectors = [m['sector'] for m in metrics.values()]
        
        fig = go.Figure()
        
        # Sort sectors by number of industries for legend ordering
        sector_counts = pd.Series(sectors).value_counts()
        sorted_sectors = sector_counts.index.tolist()
        
        for sector in sorted_sectors:
            mask = [s == sector for s in sectors]
            fig.add_trace(go.Scatter(
                x=[s for s, m in zip(strength_scores, mask) if m],
                y=[w for w, m in zip(weakness_scores, mask) if m],
                mode='markers',  # Removed text mode
                name=sector,
                marker=dict(
                    color=self.sector_colors.get(sector, '#999999'),
                    size=12,
                    symbol='circle',
                    line=dict(width=1, color='white')  # Add white border for better visibility
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>" +
                    "Sector: " + sector + "<br>" +
                    "Strength Score: %{x:.1f}<br>" +
                    "Weakness Score: %{y:.1f}<br>" +
                    "<extra></extra>"
                ),
                text=[i for i, m in zip(industries, mask) if m]  # Keep industry names for hover
            ))
        
        fig.update_layout(
            title='Industry Strength vs Weakness Matrix',
            xaxis_title='Strength Score',
            yaxis_title='Weakness Score',
            template='plotly_dark',
            height=800,
            showlegend=True,
            hovermode='closest',
            xaxis=dict(range=[0, 100]),
            yaxis=dict(range=[0, 100]),
            # Enhanced legend styling
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(0,0,0,0.1)",
                bordercolor="white",
                borderwidth=1
            ),
            # Add quadrant labels
            annotations=[
                dict(
                    x=75, y=25,
                    xref="x", yref="y",
                    text="Strong Bullish",
                    showarrow=False,
                    font=dict(color="#00FF00", size=14)
                ),
                dict(
                    x=25, y=75,
                    xref="x", yref="y",
                    text="Strong Bearish",
                    showarrow=False,
                    font=dict(color="#FF0000", size=14)
                )
            ]
        )
        
        # Add diagonal reference line
        fig.add_shape(
            type="line",
            x0=0, y0=0,
            x1=100, y1=100,
            line=dict(
                color="rgba(255, 255, 255, 0.2)",
                width=1,
                dash="dash"
            )
        )
        
        return fig

    def create_volatility_chart(self, metrics):
        """Create volatility analysis chart"""
        volatility_data = [
            {
                'industry': industry,
                'sector': m['sector'],
                'volatility': m['volatility'],
                'strength_score': m['strength_score'],
                'weakness_score': m['weakness_score'],
                'momentum': m['momentum']
            }
            for industry, m in metrics.items()
        ]
        
        # Sort by volatility
        volatility_data.sort(key=lambda x: x['volatility'], reverse=True)
        volatility_data = volatility_data[:20]  # Top 20 most volatile
        
        fig = go.Figure()
        
        # Add volatility bars
        fig.add_trace(go.Bar(
            name='Volatility',
            x=[d['industry'] for d in volatility_data],
            y=[d['volatility'] for d in volatility_data],
            marker_color=[self.sector_colors.get(d['sector'], '#999999') for d in volatility_data],
            text=[f"{v:.2f}" for v in [d['volatility'] for d in volatility_data]],
            textposition='outside',
            hovertemplate=(
                "<b>%{x}</b><br>" +
                "Volatility: %{y:.2f}<br>" +
                "<extra></extra>"
            )
        ))
        
        # Add strength/weakness/momentum markers
        max_vol = max(d['volatility'] for d in volatility_data)
        
        fig.add_trace(go.Scatter(
            name='Strength Score',
            x=[d['industry'] for d in volatility_data],
            y=[d['strength_score']/100 * max_vol for d in volatility_data],
            mode='markers',
            marker=dict(color='green', size=10, symbol='diamond'),
            hovertemplate="Strength Score: %{text:.1f}<br><extra></extra>",
            text=[d['strength_score'] for d in volatility_data]
        ))
        
        fig.add_trace(go.Scatter(
            name='Weakness Score',
            x=[d['industry'] for d in volatility_data],
            y=[d['weakness_score']/100 * max_vol for d in volatility_data],
            mode='markers',
            marker=dict(color='red', size=10, symbol='diamond'),
            hovertemplate="Weakness Score: %{text:.1f}<br><extra></extra>",
            text=[d['weakness_score'] for d in volatility_data]
        ))
        
        fig.update_layout(
            title='Industry Volatility Analysis',
            xaxis_title='Industry',
            yaxis_title='Volatility Score',
            template='plotly_dark',
            height=600,
            xaxis={'tickangle': -45},
            showlegend=True,
            barmode='overlay'
        )
        
        return fig

def main():
    st.set_page_config(page_title="Industry Analysis", layout="wide")
    
    base_dir = r"C:\Users\davet\OneDrive\kots1\Documents\trading_scripts\sector_analysis\market_rotation"
    analyzer = IndustryAnalyzer(base_dir)
    
    # Load and analyze data
    metrics, daily_data, industry_history = analyzer.load_industry_data()
    
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
            # Convert metrics to DataFrame for easy filtering
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

if __name__ == "__main__":
    main()