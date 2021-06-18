import os
import httpx
import streamlit as st


BACKEND_HOST = os.environ.get('BACKEND_HOST', '127.0.0.1:80')


@st.cache
def retrieve_results(job_id: str = None) -> httpx.Response:
    params = {'job_id': job_id} if job_id else None
    r = httpx.get(f'http://{BACKEND_HOST}/results', params=params)
    return r


image_files = st.file_uploader('Target image file',
                               type=['png', 'jpg'],
                               accept_multiple_files=True)

if len(image_files) > 0 and st.button('submit'):
    files = [('files', file) for file in image_files]

    r = httpx.post(f'http://{BACKEND_HOST}/predict', files=files)
    st.success(r.json())

r = retrieve_results()
st.write(r.json())
