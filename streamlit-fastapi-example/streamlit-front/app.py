import os
import httpx
import streamlit as st


BACKEND_HOST = os.environ.get('BACKEND_HOST', '127.0.0.1:80')


image_files = st.file_uploader('Target image file',
                               type=['png', 'jpg'],
                               accept_multiple_files=True)

if len(image_files) > 0:
    files = [('files', file) for file in image_files]

    r = httpx.post(f'http://{BACKEND_HOST}/predict', files=files)
    st.write(r)

# r = httpx.get(f'http://{BACKEND_HOST}/result')
# result_jobs = r.json()['result_jobs']
