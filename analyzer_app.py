"""Two-input local UI. Run via Start Analyzer.cmd."""
import os

import streamlit as st

from mi.analyzer.data import load_history
from mi.analyzer.engine import analyze
from mi.redact import clean

st.set_page_config(page_title='Laya Stock Analyzer', page_icon='📈', layout='centered')
st.title('Laya Stock Analyzer')
st.caption('Five-session outlook · completed daily prices · experimental prototype')


@st.cache_data(ttl=3600, show_spinner=False)
def history(symbol):
    return load_history(symbol)


with st.form('analyze'):
    a, b = st.columns(2)
    symbol = a.text_input('Ticker', 'NVDA', max_chars=15)
    price = b.number_input('Current price ($)', min_value=.0001, value=184.50, format='%.4f')
    submitted = st.form_submit_button('Analyze', type='primary', use_container_width=True)

if submitted:
    try:
        with st.spinner('Loading price history and checking the setup…'):
            data = history(symbol.strip().upper())
            models = tuple(x.strip() for x in os.getenv('ANALYZER_MODELS', 'linear').split(',') if x.strip())
            result = analyze(symbol, price, data.data, models).to_dict()
        st.session_state['result'] = (result, data)
    except Exception as exc:
        st.session_state.pop('result', None)
        st.error('Analysis unavailable. ' + str(clean(str(exc))))

if 'result' in st.session_state:
    result, data = st.session_state['result']
    st.subheader(f"{result['ticker']} · {result['action']}")
    a, b, c = st.columns(3)
    a.metric('Direction', result['direction'])
    b.metric('Setup quality', result['setup_quality'])
    c.metric('Reversal', result['reversal'])
    st.write('**Breakout:** ' + result['breakout'])
    if result['entry'] is not None:
        cols = st.columns(5)
        for col, label, value in zip(cols, ['Entry reference', 'Stop', 'Target 1', 'Target 2', 'Target 3'],
                                     [result['entry'], result['stop'], *result['targets']]):
            col.metric(label, f'${value:,.2f}')
        st.caption('Levels use your entered price and 1.5× daily ATR risk; targets are 1R, 2R, 3R. No orders are placed.')
    else:
        st.info('Wait — no active entry, stop, or targets.')
    st.caption(f"Daily history through {result['as_of']} · Source: {data.sources()['close']} · Entered price: ${result['current_price']:,.2f}")
    st.caption('Models used: ' + ', '.join(result['components']))
    for warning in result['warnings']:
        st.warning(warning)
    with st.expander('Evidence and model details'):
        st.line_chart(data.data.close.tail(180))
        st.json(result)
        st.dataframe(data.audit(), hide_index=True)
    st.download_button('Save analysis', data=__import__('json').dumps(clean(result), indent=2),
                       file_name=f"{result['ticker']}-analysis.json", mime='application/json')

st.caption('Research prototype. Model agreement and setup labels have not demonstrated a reliable trading edge.')
