import pandas as pd
from plot_preds_vs_actuals import plot_preds_vs_actuals
from mk_predictions import predict
import streamlit as st

# set_page_config() can only be called once per app, and must be called as the 
# first Streamlit command in your script.
st.set_page_config(page_title='ROI-Prediction-Univ3-WETH-GMX', 
    layout='wide', page_icon='📈')

st.title('Predict ROI and Fee APR of UniV3 WETH-GMX positions on Arbitrum with Machine Learning')

# plot actual vs. predictions on test set
st.header('Model performance on data not used in training')
c1, c2 = st.columns(2)
with c1:
    fig = plot_preds_vs_actuals('roi')
    st.plotly_chart(fig, use_container_width=True)
    st.write('ROI is the total PnL divided by initial deposits. It accounts for gas and impermanent loss.')
with c2:
    fig = plot_preds_vs_actuals('fee_apr')
    st.plotly_chart(fig, use_container_width=True)
    st.write('Fee APR is the annualized rate of fees earned. It does NOT include gas or impermanent loss.')

st.markdown("""---""")

# allow user to input values and get predictions
st.header('Predict by entering your own values')
st.markdown("Make sure enter numbers as precise as possible. For example, a price range of 28.1 - 39.8 over 14.0 days can give a very different prediction than 28.1224 - 39.8142 over 14.6 days.")

# user input
c1, c2, c3, c4 = st.columns(4)
with c1:
    fee_tier = st.selectbox("Select fee-tier:", ("1.0%", "0.3%"))
with c2:
    price_lower = st.number_input(
        "Set Price Range - Min", value=28.0, 
        min_value=4.0, max_value=42.0,
        step=0.0001, format="%.4f")
with c3:
    price_upper = st.number_input(
        "Set Price Range - Max", value=40.0,
        min_value=23.0,
        max_value=365.0,
        step=0.0001, format="%.4f")
with c4:
    age = st.number_input(
        "How many days will you provide liquidity for?", value=7.0, 
        min_value=3/24, max_value=132.0, step=0.1, format="%.1f")

# predict using ML models
if price_upper <= price_lower:
    st.write('Make sure your upper price limit > lower price limit!')
else:
    row_pred = pd.DataFrame({
        'ROI': predict('roi', fee_tier, price_lower, price_upper, age),
        'Fee APR': predict('log1p_fee_apr', fee_tier, price_lower, price_upper, age), 
    }, index=['Prediction'])
    
    def color(s):
        # color the cell value of a data frame green if it's > 0 
        # and red otherwise
        res = []
        for val in s:
            if val > 0:
                res.append('color: #8db600') 
            else: 
                res.append('color: #ff2052')
        return res 
    st.header('Output')
    st.table(row_pred.style.apply(color, axis=1).format('{:.2%}')\
        .set_properties(**{'font-size': '25px'}))

st.markdown("""---""")

# method description
st.header('Methodology')
c1, c2 = st.columns(2)
with c1:
    st.subheader('Data')
    st.markdown("Ideally, we want to work with a dataset that has only\n1. positions that never had liquidity partially removed,\n2. positions that never had additional liquidity added,\n3. positions that ran its course and had 100% liquidity removed and no new liquidity added afterwards.\nTwo datasets of 1 and 2 were downloaded from Revert Finance on Nov 23, 2022 at 23:08:00 UTC and on Feb 09, 2023 at 01:28:00 UTC. The data-pulling script was written by [0x1egolas](https://twitter.com/0x1egolas). Unfortunately, data of 3 cannot be easily obtained.")
    st.markdown("We then took a subset of data that meet all of the following criteria:\n - initial deposit value is between \$50 and \$1M US Dollars,\n- age of position is at least 3 hours,\n- lower ETHGMX limit is at least 4,\n- upper ETHGMX limit is at most 400.")
    st.markdown("Ranges of Important Variables on this subset:\n- Age ranges from 0.15 to 191.202 days,\n- Deposit ranges from \$51 to \$786,878 USD,\n- Price lower limit ranges from 5.1 to 41.3,\n- Price upper limit ranges from 22.2 to 364.9,\n- ROI ranges from -22.45% to 32.72%,\n- Fee APR ranges from 3.03% to 2657.16%.")
    st.markdown("Finally, we split this subset into a training set (80%, 561 records) and a test set (20%, 141 records). We trained ML models on the training set with cross-validation (cv). Throughout the training, we did not touch the test set, nor did we once peek at model performance on the test set. We chose the best model according to cv error. Only after we finalized the model, we used it on the test set to get the predictions displayed at the top of this page.")
with c2:
    st.subheader('Machine Learning')
    st.markdown("Targets:\n- ROI \n- Fee APR")
    st.markdown("Features:\n- Fee Tier \n- Age \n- Price Lower Limit \n- Price Upper Limit \n- Price Range (Price Upper Limit - Price Lower Limit).")
    st.markdown("Transformations:\n- Applied `log1p()` to Fee APR, Age, Price Upper Limit and Price Range because they had long right tails.\n- Standardized Price Lower Limit.\n- One-Hot Encoded Fee Tier.")
    st.markdown("Model selection:\n- XGBOOST regressor was trained, where a single parameter, learning rate, was tuned for optimal performance via 5-fold cross-validation, with negative mean absolute error as the scoring metric.\n- The best model for ROI prediction had a CV error of 2%.\n- The best model for Fee APR prediction had a CV error of 17.7%, which isn't big given the fee apr in the sample ranges from 3.03% to 2657.16%.")
    st.markdown("Test Set Performance:\n- The best model for ROI prediction had a test error of 2%.\n- The best model for Fee APR prediction had a test error of 18.1%.")

st.markdown("""---""")
    
# about
c1, c2 = st.columns(2)
with c1:
    st.subheader('Code')
    st.markdown('- [Download data](https://github.com/coindataschool/univ3lp/blob/main/scripts/01-pull-data.py)')
    st.markdown('- [Prepare data](https://github.com/coindataschool/univ3lp/blob/main/scripts/02-prep-data.py)')
    st.markdown('- [Train Models](https://github.com/coindataschool/univ3lp/blob/main/scripts/03-model-roi-n-fee-apr-weth-gmx.py)')
    st.markdown('- [This Dashboard](https://github.com/coindataschool/univ3-roi-prediction-weth-gmx)')
with c2:
    st.subheader('Support my work')
    st.markdown("- Buy our [UniV3 Guide](https://twitter.com/coindataschool/status/1613886358938058752)")
    st.markdown("- Send me ETH: `0x783c5546c863f65481bd05fd0e3fd5f26724604e`")
    st.markdown("- [Tip me sat](https://tippin.me/@coindataschool)")
    st.markdown("- [Buy me a coffee](https://ko-fi.com/coindataschool)")
    st.markdown("- Subscribe to my [newsletter](https://coindataschool.substack.com/about)")
    st.markdown("- Follow me on twitter: [@coindataschool](https://twitter.com/coindataschool)")
    st.markdown("- Follow me on github: [@coindataschool](https://github.com/coindataschool)")
    st.markdown("- Find me on Dune: [@coindataschool](https://dune.com/coindataschool)")