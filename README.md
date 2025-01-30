# market-sector-analysis
Industry and sector analysis for stock market trends
# Industry Analysis Visualization Documentation

## 1. Strength Score Analysis

### Components and Weights
- 30% Addition Ratio = New stocks added / Total volume
- 30% Current Strength = Current active stocks / Historical maximum stocks
- 40% Recent Trend = Positive changes in last 7 days / Historical maximum
- Volatility Impact: All scores are reduced by volatility factor (1 - volatility * 0.15)

### Calculations
```python
addition_ratio = total_added / (total_added + total_removed)
current_strength = avg_active / max_active
trend_strength = max(recent_trend, 0) / max_active

raw_strength_score = (
    (addition_ratio * 0.3) +
    (current_strength * 0.3) +
    (trend_strength * 0.4)
)
final_strength_score = raw_strength_score * volatility_factor
```

## 2. Weakness Score Analysis

### Components and Weights
- 30% Removal Ratio = Stocks removed / Total volume
- 30% Current Weakness = 1 - (Current active / Historical maximum)
- 40% Recent Trend = Absolute value of negative changes / Historical maximum
- Same volatility reduction as strength score

### Calculations
```python
removal_ratio = total_removed / (total_added + total_removed)
current_weakness = 1 - (avg_active / max_active)
trend_weakness = abs(min(recent_trend, 0)) / max_active

raw_weakness_score = (
    (removal_ratio * 0.3) +
    (current_weakness * 0.3) +
    (trend_weakness * 0.4)
)
final_weakness_score = raw_weakness_score * volatility_factor
```

## 3. Momentum Analysis

### Components
- Daily Changes: Net change in stocks (added - removed) each day
- Momentum Score: Weighted average of recent changes
- Momentum Change: Rate of change in momentum

### Calculations
```python
# For each day:
net_change = stocks_added - stocks_removed

# 5-day momentum with exponential weighting
weights = np.exp(np.linspace(0, 1, len(recent_changes)))
momentum = np.average(recent_changes, weights=weights)

# Momentum change
momentum_change = current_momentum - momentum_scores[-5]
```

The exponential weighting means recent changes have more impact than older ones. For example, with 5 days of data:
- Day 5 (most recent): ~40% weight
- Day 4: ~25% weight
- Day 3: ~16% weight
- Day 2: ~11% weight
- Day 1: ~8% weight

## 4. Volatility Analysis

### Components
- Volatility Score: Standard deviation of daily changes
- Impact on Scores: Reduces both strength and weakness scores
- Volatility Factor: 1 - (volatility * 0.15)

### Calculations
```python
volatility = np.std(daily_changes)
volatility_factor = max(0, 1 - (volatility * 0.15))
```

## 5. Comparison Matrix (Scatter Plot)

### Components
- X-axis: Strength Score (0-100)
- Y-axis: Weakness Score (0-100)
- Diagonal Line: Equal strength/weakness
- Quadrants:
  - Top Left: Strong Bearish (High weakness, Low strength)
  - Bottom Right: Strong Bullish (High strength, Low weakness)

### Interpretation
- Distance from diagonal indicates trend strength
- Points above diagonal: More bearish
- Points below diagonal: More bullish
- Color coding by sector shows sector-wide trends

## 6. Daily Data Processing
- Files are processed chronologically
- Missing days are handled with zero momentum
- Stock additions/removals are tracked per industry
- Historical maximums are maintained for normalization
- Data gaps are filled with neutral values to maintain continuity

## Use Cases
1. **Trend Identification**: Strength/Weakness charts show strongest trends
2. **Sector Rotation**: Comparison matrix shows sector movements
3. **Risk Assessment**: Volatility analysis highlights stable vs unstable industries
4. **Momentum Trading**: Momentum chart identifies accelerating trends
5. **Industry Health**: Active stocks count and volume trends indicate industry vitality
