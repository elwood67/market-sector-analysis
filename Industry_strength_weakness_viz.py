import streamlit as st
import pandas as pd
import json
from pathlib import Path
import plotly.graph_objects as go
import os

def check_password():
    """Returns `True` if the user had the correct password."""
    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", key="password", on_change=password_entered)
        return False
    return st.session_state["password_correct"]

def password_entered():
    if st.session_state["password"] == st.secrets["password"]:
        st.session_state["password_correct"] = True
    else:
        st.session_state["password_correct"] = False

class SimpleIndustryAnalyzer:
    def __init__(self):
        self.base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.data_sets_dir = self.base_dir / 'data' / 'datasets'
        self.weights_path = self.base_dir / 'data' / 'weighting' / 'industry_weights.json'
        
        # Load weights first
        self.industry_weights = self.load_industry_weights()
        
        # Color scheme
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

    def load_industry_weights(self):
        """Load industry weighting data"""
        try:
            with open(self.weights_path, 'r') as f:
                weights_data = json.load(f)
                return weights_data['weights']
        except Exception as e:
            st.warning(f"Could not load industry weights: {e}")
            return {}

    def analyze_industry_data(self):
        """Load and analyze industry data"""
        industry_metrics = {}
        
        # Get most recent data file
        json_files = sorted(list(self.data_sets_dir.glob('comparison_date_*.json')))
        if not json_files:
            st.error("No data files found!")
            return {}
        
        # Track active stocks per industry
        active_stocks = {}  # industry -> set of active stocks
        total_stocks = {}   # industry -> total stocks ever seen
        
        # Process all files to get historical context
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            for change in data['changes']:
                industry = change.get('industry')
                sector = change.get('sector')
                status = change['status']
                ticker = change['ticker']
                
                if sector != 'Unknown' and industry:
                    # Initialize tracking for new industries
                    if industry not in active_stocks:
                        active_stocks[industry] = set()
                        total_stocks[industry] = set()
                    
                    # Track all stocks we've seen
                    total_stocks[industry].add(ticker)
                    
                    # Update active stocks
                    if status == 'added':
                        active_stocks[industry].add(ticker)
                    elif status == 'removed':
                        active_stocks[industry].discard(ticker)
        
        # Calculate metrics for each industry
        for industry in active_stocks:
            if industry in self.industry_weights:
                weight = self.industry_weights[industry]['composite_weight']
                sector = self.industry_weights[industry]['sector']
                
                # Calculate strength based on percentage of bullish stocks
                total_count = len(total_stocks[industry])
                active_count = len(active_stocks[industry])
                
                if total_count > 0:
                    bullish_percentage = active_count / total_count
                    # Adjust the percentage by the industry weight
                    weighted_strength = bullish_percentage * weight * 100
                    weighted_weakness = (1 - bullish_percentage) * weight * 100
                    
                    industry_metrics[industry] = {
                        'sector': sector,
                        'active_stocks': active_count,
                        'total_stocks': total_count,
                        'bullish_percentage': bullish_percentage,
                        'weight': weight,
                        'strength_score': weighted_strength,
                        'weakness_score': weighted_weakness
                    }
        
        return industry_metrics

    def create_strength_chart(self, metrics):
        """Create strength analysis visualization"""
        # Sort industries by strength score
        sorted_industries = sorted(
            metrics.items(),
            key=lambda x: x[1]['strength_score'],
            reverse=True
        )
        
        industries = []
        scores = []
        colors = []
        hover_text = []
        
        for industry, m in sorted_industries[:30]:  # Top 30 industries
            industries.append(industry)
            scores.append(m['strength_score'])
            colors.append(self.sector_colors.get(m['sector'], '#999999'))
            
            hover_text.append(
                f"Industry: {industry}<br>" +
                f"Sector: {m['sector']}<br>" +
                f"Active/Total Stocks: {m['active_stocks']}/{m['total_stocks']}<br>" +
                f"Bullish %: {m['bullish_percentage']*100:.1f}%<br>" +
                f"Industry Weight: {m['weight']:.4f}<br>" +
                f"Strength Score: {m['strength_score']:.1f}"
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
        
        fig.update_layout(
            title='Industry Strength Analysis (Size Weighted)',
            xaxis_title='Strength Score',
            yaxis_title='Industry',
            template='plotly_dark',
            height=800,
            margin=dict(l=200, r=100)
        )
        
        return fig

def main():
    st.set_page_config(page_title="Industry Analysis", layout="wide")
    
    if not check_password():
        st.stop()
    
    analyzer = SimpleIndustryAnalyzer()
    
    try:
        # Load and analyze data
        metrics = analyzer.analyze_industry_data()
        
        if not metrics:
            st.error("No data was loaded. Please check your data directory.")
            st.stop()
        
        # Display strength chart
        st.title("Industry Analysis")
        
        # Add explanation of calculations
        st.markdown("""
        ### Strength Score Calculation:
        - Base strength = Percentage of stocks that are bullish (active/total)
        - Final score = Base strength × Industry weight
        - Industry weights consider market impact and size category
        """)
        
        # Show the chart
        st.plotly_chart(analyzer.create_strength_chart(metrics), use_container_width=True)
        
        # Show raw data table for verification
        st.subheader("Raw Data (for verification)")
        df = pd.DataFrame([
            {
                'Industry': industry,
                'Sector': m['sector'],
                'Active Stocks': m['active_stocks'],
                'Total Stocks': m['total_stocks'],
                'Bullish %': f"{m['bullish_percentage']*100:.1f}%",
                'Weight': f"{m['weight']:.4f}",
                'Strength Score': f"{m['strength_score']:.1f}"
            }
            for industry, m in metrics.items()
        ])
        st.dataframe(df.sort_values('Strength Score', ascending=False))
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()