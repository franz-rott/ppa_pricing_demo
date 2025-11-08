# Solar PPA Pricing Analysis with Battery Storage Integration

## Overview
Analysis of synthetic Power Purchase Agreement (PPA) pricing for a 20 MW solar installation, comparing scenarios with and without a 10 MW / 20 MWh battery storage system.

### Key Features
- 🔄 Training/pricing period: 2022-2025
- 📊 PnL projection period: 2026-2029
- 🔋 Battery storage analysis: 10 MW / 20 MWh
- 📈 Interactive risk parameter adjustment
- 💹 Comprehensive PnL projections

## Project Structure
```
ppa_demo/
├── data/                    # Synthetic data storage
├── figures/                 # Generated plots
├── notebooks/
│   └── 01_demo_solar_ppa.ipynb  # Main analysis notebook
└── src/
    ├── battery.py          # Battery simulation logic
    ├── config.py           # Configuration parameters
    ├── features.py         # Feature engineering
    ├── plots.py           # Visualization functions
    ├── pnl.py             # PnL calculations
    ├── pricing.py         # PPA pricing logic
    ├── risk_premia.py     # Risk premium calculations
    ├── synth_data.py      # Synthetic data generation
    └── utils.py           # Utility functions
```

## Installation

### Prerequisites
- Python 3.10+
- Jupyter Lab/Notebook

### Setup
1. Clone the repository:
   ```bash
   git clone [your-repo-url]
   cd ppa_demo
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start Jupyter Lab:
   ```bash
   jupyter lab
   ```

2. Open `notebooks/01_demo_solar_ppa.ipynb`

3. Run all cells to:
   - Generate synthetic data (first run only)
   - View interactive analysis
   - Explore different scenarios

### Interactive Controls
- Cost Parameters (€/MWh)
  - Base costs
  - Margin
  - Other risks

- Risk Parameters (λ)
  - Market risk
  - Profile risk
  - Volatility risk

- Battery Parameters
  - Round-trip efficiency
  - Dispatch thresholds (p_low, p_high)
  - Error and spread scaling

## Key Metrics

- 📊 **Solar Capture Factor**: ~0.90 (2022-2025)
- 💰 **PnL Scenarios**:
  - Without Battery: Positive returns
  - With Battery: Enhanced returns with modified risk profile
- ⚡ **Battery Impact**:
  - Price arbitrage opportunities
  - Imbalance cost reduction
  - Risk profile modification

## Technical Notes

- All data is synthetic but calibrated to realistic market patterns
- Timezone: Europe/Berlin
- Hourly resolution
- Fixed random seed for reproducibility
- PPA pricing formula:
  `Fix = Expected Market Price × Solar Profile Factor − Risk Premia − Costs − Margin − Other risks`

