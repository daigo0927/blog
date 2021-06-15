import os
import httpx
import streamlit as st


ML_SERVER_HOST = os.environ.get('ML_SERVER_HOST', '127.0.0.1:80')


image_files = st.file_uploader('Target image file',
                               type=['png', 'jpg'],
                               accept_multiple_files=True)

if len(image_files) > 0:
    files = [('files', f.getvalue()) for f in image_files]

    origin = f'http://{ML_SERVER_HOST}'
    r = httpx.post(f'{origin}/predict', files=files)
    st.write(r)

