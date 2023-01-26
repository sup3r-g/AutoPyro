import streamlit as st

st.json({
    'foo': 'bar',
    'baz': 'boz',
    'stuff': [
        'stuff 1',
        'stuff 2',
        'stuff 3',
        'stuff 5',
    ],
})

A = st.slider('A', 0, 130, 25)
K = st.slider('K', 0, 130, 25)
B = st.slider('B', 0, 130, 25)
Q = st.slider('Q', 0, 130, 25)
C = st.slider('C', 0, 130, 25)
M = st.slider('M', 0, 130, 25)

st.write('Значения параметров:', A, K, B, Q, C, M)